import json
import os
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
import torch.onnx
from torchgen import model
import torchvision.transforms.v2 as transforms  # https://pytorch.org/vision/main/transforms.html
from loguru import logger
from omegaconf import OmegaConf
from torch import nn
from torch import optim as optim

from ready.models.deeplabv3 import DeepLabV3, deeplabv3_mobilenet_v3_large
from ready.utils.datasets import MobiousDataset
from ready.utils.metrics import evaluate
from ready.utils.utils import (HOME_PATH, create_data_loaders, evaluate_model,
                               loss_values_file_writer,
                               performance_file_writer,
                               sanity_check_trainloader, set_data_directory,
                               test_accuracy_file_writer, training_loop,
                               validation_loop)

torch.cuda.empty_cache()


def save_checkpoint(state, path):
    """
    Save checkpoint method
    """
    torch.save(state, path)
    print("Checkpoint saved at {}".format(path))


def norm_image(hot_img):
    """
    Normalise image
    """
    return torch.argmax(hot_img, 0)

def calculate_epoch_loss(running_loss, num_samples):
    """
    Calculate epoch loss
    """

    return running_loss/num_samples

def main(args):
    """
    Train pipeline for DEEPLABV3 with Mobious dataset

    #CHECK epoch = None
    #CHECK if weight_fn is not None:
    #CHECK add checkpoint
    #CHECK add execution time
    """
    #load and read .yaml objects and variables
    config_file = args.config_file
    config = OmegaConf.load(config_file)

    # Import all arguments from config
    DATA_PATH = config.dataset.data_path
    MODEL_PATH = config.dataset.models_path
    GITHUB_DATA_PATH = config.dataset.github_data_path
    debug_print_flag = config.model.debug_print_flag
    use_github_data_path_flag = config.dataset.use_github_data_path_flag

    TRANSFORM_OPERATION = config.transforms.transform_operation
    TARGET_TRANSFORM_OPERATION = config.transforms.target_transform_operation

    TRAIN_SET_RATIO = config.datasets_splitting_ratios.train_set
    VALIDATION_SET_RATIO = config.datasets_splitting_ratios.validation_set
    TEST_SET_RATIO = config.datasets_splitting_ratios.test_set

    PRETRAINED_MODEL_FOLDER = config.pretrained_model.models_folder_path
    CHECKPOINT_PATH = config.pretrained_model.checkpoint_path
    MODEL_NAME_FOR_EVAL = config.pretrained_model.model_name_for_eval
    evaluation_with_pretrained_model_flag = config.pretrained_model.evaluation_with_pretrained_model_flag

    #load hperparameters from connfig
    batch_size = config.model_hyperparameters.batch_size
    num_workers = config.model_hyperparameters.num_workers
    learning_rate = config.model_hyperparameters.learning_rate
    run_epoch = config.model_hyperparameters.epochs
    
    SEED = 42

    FULL_DATA_PATH = os.path.join(Path.home(), DATA_PATH)
    FULL_GITHUB_DATA_PATH = os.path.join(Path.cwd(), GITHUB_DATA_PATH)
    FULL_MODEL_PATH = os.path.join(Path.home(), MODEL_PATH)
    FULL_PRETRAINED_MODEL_PATH = os.path.join(Path.home(), PRETRAINED_MODEL_FOLDER)
    if not os.path.exists(FULL_MODEL_PATH):
        os.makedirs(FULL_MODEL_PATH, exist_ok=True)

    #set to true to use samole data from repo, false to point to one on local computer (actual dataset)
    data_path = FULL_GITHUB_DATA_PATH if use_github_data_path_flag else FULL_DATA_PATH

    starttime = time.time()  # print(f'Starting training loop at {startt}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0)[0:20] if torch.cuda.is_available() else "cpu"
    device_name= device_name.replace (" ", "_")
    logger.info(f"GPU DEVICE NAME: {device_name}")
    cuda_available = torch.cuda.is_available()

    #Transforms for images
    transforms_img = transforms.Compose([
                                            transforms.ToImage(),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomVerticalFlip(p=0.5),
                                            transforms.RandomRotation(45),
                                            transforms.GaussianBlur(kernel_size=(5, 13), sigma=(1, 50)),
                                            transforms.Normalize(mean=[0.285, 0.456, 0.406], std=[0.529, 0.524, 0.525]),
                                            transforms.ElasticTransform(alpha=100.0, sigma=5.0),
                                            transforms.Resize((128, 128)),      #set to 128x128 to test data resizing on model, same with mask
                                            ])
     #Transforms for masks
    transforms_rotations = transforms.Compose([
                                            transforms.ToImage(),
                                            transforms.RandomHorizontalFlip(p=0.5),
                                            transforms.RandomVerticalFlip(p=0.5),
                                            transforms.RandomRotation(45),
                                            transforms.Resize((128, 128)),
                                            ])
    
    #if none in config file, no transformartions apply
    transform_map = {
    'transforms_img': transforms_img,
    'transforms_rotations': transforms_rotations
    }

    transform_arg = transform_map.get(TRANSFORM_OPERATION,None)
    target_transform_arg = transform_map.get(TARGET_TRANSFORM_OPERATION, None)

  
    full_dataset = MobiousDataset(
        data_path, transform=transform_arg ,target_transform=target_transform_arg
        )

    data_splitting_ratios = [TRAIN_SET_RATIO, VALIDATION_SET_RATIO, TEST_SET_RATIO]

    train_loader, validation_loader, test_loader = create_data_loaders(full_dataset=full_dataset,
                                                                       data_splitting_ratios=data_splitting_ratios,
                                                                       seed=SEED,
                                                                       batch_size=batch_size,
                                                                       num_workers=num_workers)
    
    if debug_print_flag:
        sanity_check_trainloader(train_loader, cuda_available)

    current_time_stamp= datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
    PATH = FULL_MODEL_PATH+"/"+ current_time_stamp + "_" + device_name

    model = DeepLabV3(nch_out=4)

    if not evaluation_with_pretrained_model_flag:
        #check nuber of parameters of deeplab model (APPX. 11M)
        num_params = len(nn.utils.parameters_to_vector(model.parameters()))
        logger.info(f"Number of parameters in model: {num_params}")

        #weight decay to prevent overfitting of model and ADAMW optimizer to improve generalization of model
        optimizer = optim.AdamW(model.parameters(),lr= learning_rate, weight_decay=1e-4)  
        loss_fn = nn.CrossEntropyLoss()

        if cuda_available:
            model.cuda()
            loss_fn.cuda()

        epoch = None
        
        #metrics to be calculated for each epoch
        performance_metrics_labels = ["accuracy",
                                      "f1",
                                      "recall",
                                      "precision",
                                      "fbeta",
                                      "miou",
                                      "dice",
                                      "hausdorff_distance",  #It measures the maximum distance between the predicted and ground truth boundaries, providing insight into the worst-case segmentation error.
                                      ]
        # Training Metrics
        training_loss_values = []
       
        # Validation metrics
        validation_loss_values = []
        

        logger.info("Commencing Training and Validation Loop")
        logger.info(f"#########################")

        #resets the preformace for every epoch to 0
        for i in range(epoch + 1 if epoch is not None else 1, run_epoch + 1):
            logger.info(f"Training and Validation loop at Epoch: {i} out of {run_epoch}")
            total_training_running_loss, total_validation_running_loss = 0.0, 0.0
            total_num_training_samples, total_num_validation_samples= 0, 0
           

            training_performance = {
            "accuracy": 0.0,
            "f1": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "fbeta": 0.0,
            "miou": 0.0,
            "dice": 0.0,
            "hausdorff_distance": 0.0, 
            #Good mIoU but bad Hausdorff = model gets most pixels right but has some bad boundary errors
            #Good Hausdorff = boundary is accurate, clinically useful
        }
            
            validation_performance = {
            "accuracy": 0.0,
            "f1": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "fbeta": 0.0,
            "miou": 0.0,
            "dice": 0.0,
            "hausdorff_distance": 0.0,
        }
            # Training
            logger.info(f"Training Section")
            for j, data in enumerate(train_loader, 1):
                current_training_loss, num_samples_processed = training_loop(model=model,
                              current_idx=j,
                              current_data=data,
                              optimizer=optimizer,
                              training_performance_dict=training_performance,
                              loss_fn = loss_fn,
                              cuda_available=cuda_available)

                total_training_running_loss += current_training_loss
                total_num_training_samples += num_samples_processed

            logger.info(f"#########################")

            # Validation
            logger.info("Validation Section")
            model.eval()    #bath normalization to ensure model's predictions are not affected by the batch statistics during validation and are stable and consistent
            with torch.set_grad_enabled(False):
                     for j, data in enumerate(validation_loader, 1):
                        current_validation_loss, num_samples_processed = validation_loop(model=model,
                                        current_idx=j,
                                        current_data=data,
                                        optimizer=optimizer,
                                        validation_performance_dict=validation_performance,
                                        loss_fn=loss_fn,
                                        cuda_available=cuda_available)
                        

                        total_validation_running_loss += current_validation_loss
                        total_num_validation_samples += num_samples_processed
            model.train()

            #Calculate average loss per image in each epoch for training and validation

            training_epoch_loss = calculate_epoch_loss(total_training_running_loss, total_num_training_samples)
            validation_epoch_loss = calculate_epoch_loss(total_validation_running_loss, total_num_validation_samples)

            training_loss_values.append(training_epoch_loss)
            validation_loss_values.append(validation_epoch_loss)
            print(f"\nTraining epoch loss: {training_epoch_loss:.4f}")
            print(f"Validation epoch loss: {validation_epoch_loss:.4f}\n")

            print(f"Training Metrics:")
            for key in performance_metrics_labels:
                training_performance[key] /= total_num_training_samples
                print(f"Average {key} @ epoch: {training_performance[key]:.4f}")

            print(f"\nValidation Metrics:")
            for key in performance_metrics_labels:
                validation_performance[key] /= total_num_validation_samples
                print(f"Average {key} @ epoch: {validation_performance[key]: .4f}")

        logger.info(f"#########################")
        logger.info(f"Training and Validation complete.")

        PATH = FULL_MODEL_PATH+"/"+ current_time_stamp + "_" + device_name
        if not os.path.exists(PATH):
            os.makedirs(PATH, exist_ok=True)

        #save model weights and performace metrics to a .pth file
        if not debug_print_flag:
            model_name = PATH+"/weights_" + current_time_stamp + ".pth"
            torch.save(model.state_dict(), model_name)
            logger.info(f"Saved PyTorch Model State to {model_name}")

            performance_file_prefix_to_performance_dict = {
                "/training_performance_" : training_performance,
                "/validation_performance_" : validation_performance
            }

            logger.info(f"Writing performance metrics to {PATH}")
            # Write performance metrics to .json files
            for file_prefix, performance_dict in performance_file_prefix_to_performance_dict.items():
                performance_file_writer(folder_path=PATH, file_prefix=file_prefix, performance_dict=performance_dict, current_time_stamp=current_time_stamp)

            logger.info(f"#########################")

            # To create a plot showing how loss values change for every epoch,
            # use src/ready/apis/plot_losses.py script.

            logger.info(f"Writing loss values to {PATH}")

            loss_values_prefix_to_loss_value = {
                "/training_loss_values_" : training_loss_values,
                "/validation_loss_values_" : validation_loss_values
            }

            for file_prefix, loss_values in loss_values_prefix_to_loss_value.items():
                loss_values_file_writer(folder_path=PATH, file_prefix=file_prefix, loss_values=loss_values, current_time_stamp=current_time_stamp)

        else:
            logger.info(f"Model saving is disabled, set debug_print_flag to False (-df 0) to save model")

        endtime = time.time()
        elapsedtime = endtime - starttime
        logger.info(f"Elapsed time for the training and validation loop: {elapsedtime} (sec)")

        logger.info("Commencing Evaluation")

        #runs the now trained model on the 10% test data
        test_accuracy = evaluate_model(model=model, test_loader=test_loader, device=device)
        test_accuracy_file_writer(folder_path=PATH, test_accuracy=test_accuracy,
                                  current_time_stamp=current_time_stamp, pretrained_model_flag=evaluation_with_pretrained_model_flag)

    else:

        logger.info("Skipping Training and Validation Loop. Straight to Evaluation.")

        time_of_testing = datetime.now().strftime("%d-%b-%Y_%H_%M_%S")

        folder_containg_pretrained_model = MODEL_NAME_FOR_EVAL[8:-4] + "_" + device_name
        model_path = os.path.join(FULL_PRETRAINED_MODEL_PATH, folder_containg_pretrained_model, MODEL_NAME_FOR_EVAL)

        model.load_state_dict(torch.load(model_path, weights_only=True))
        # model.eval()
        PATH = FULL_MODEL_PATH+"/"+ folder_containg_pretrained_model
        if not os.path.exists(PATH):
            os.makedirs(PATH, exist_ok=True)

        test_accuracy = evaluate_model(model=model, test_loader=test_loader,
                                       device=device)

        test_accuracy_file_writer(folder_path=PATH, test_accuracy=test_accuracy, current_time_stamp=time_of_testing, pretrained_model_flag=evaluation_with_pretrained_model_flag)

    logger.info(f"Model Accuracy: {test_accuracy: .4f}%")
    logger.info(f"Completed!")

if __name__ == "__main__":
    parser = ArgumentParser(description="READY demo application.")
    parser.add_argument("-c", "--config_file", help="Config filename with path", type=str)

    args = parser.parse_args()
    main(args)
