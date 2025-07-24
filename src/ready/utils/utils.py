"""
utils
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ready.utils.metrics import evaluate

HOME_PATH = Path.home()
REPOSITORY_PATH = Path.cwd()


def set_data_directory(main_path: str = None, data_path: str = None):
    """
    set_data_directory with input variable:
        data_path.
        For example:
        set_data_directory("data/mobious/sample-frames/test640x400")
    """
    if main_path is None:
        main_path = REPOSITORY_PATH
    print(f"main_path: {main_path}")
    print(f"data_path: {data_path}")
    os.chdir(os.path.join(main_path, data_path))


def sanity_check(trainloader, neural_network, cuda_available):
    """
    Sanity check of trainloader for openEDS
    #TODO Sanity check for RTI-eyes datasets?
    """
    # f, axarr = plt.subplots(1, 3)

    for images, labels in trainloader:
        if cuda_available:
            images = images.cuda()
            labels = labels.cuda()

        # print(images[0].unsqueeze(0).size()) #torch.Size([1, 1, 400, 640])
        outputs = neural_network(images[0].unsqueeze(0))
        # print("nl", labels[0], "no", outputs[0])
        print(
            f"   CHECK images[0].shape: {images[0].shape}, \
                labels[0].shape: {labels[0].shape}, outputs.shape: {outputs.shape}"
        )
        # nl = norm_image(labels[0].reshape([400, 640, 4]).
        # swapaxes(0, 2).swapaxes(1, 2)).cpu().squeeze(0)
        no = norm_image(outputs[0]).cpu().squeeze(0)
        print(
            f"   CHECK no[no == 0].size(): {no[no == 0].size()}, \
                no[no == 1].size(): {no[no == 1].size()}, no[no == 2].size(): \
                    {no[no == 2].size()}, no[no == 3].size(): {no[no == 3].size()}"
        )

        # TOSAVE_PLOTS_TEMPORALY?
        #
        # axarr[0].imshow((images[0] * 255).to(torch.long).squeeze(0).cpu())
        # print("NLLLL", nl.shape)
        # axarr[1].imshow(labels[0].squeeze(0).cpu())
        # axarr[2].imshow(no)

        # plt.show()

        break


def sanity_check_trainloader(trainloader, cuda_available):
    """
    Sanity check of trainloader
    """
    # f, axarr = plt.subplots(1, 3)

    print(f"############################")
    print(f"# Sanity check of trainloader")
    print(f"# trainloader.batch_size: {trainloader.batch_size}")

    for images, labels in trainloader:
        if cuda_available:
            images = images.cuda()
            labels = labels.cuda()

        print(f"# images.size() {images.size()};\
        type(images): {type(images)};\
        images.type: {images.type()} ")
        # images.size() torch.Size([5, 3, 400, 640])
        print(f"# labels.size() {labels.size()};\
        type(labels): {type(labels)};\
        labels.type: {labels.type()} ")
        # labels.size() torch.Size([5, 400, 640]);


        if cuda_available:
            images = images.cuda()
            labels = labels.cuda()
            # images
            print(f"# images.size() {images.size()};\
            type(images): {type(images)};\
            images.type: {images.type()} ")
            # torch.Size([batch_size_, 3, 400, 640]);
            # <class 'torch.Tensor'>;
            # torch.cuda.FloatTensor
            # labels
            print(f"# labels.size() {labels.size()};\
            type(labels): {type(labels)};\
            labels.type: {labels.type()} ")
            # torch.Size([batch_size_, 400, 640]),
            # <class 'torch.Tensor'>, torch.cuda.LongTensor


        #TODO add sanity check for plotting image and encoded masks
        plt.subplot(2,1,1), plt.imshow(images[0].cpu().permute(1,2,0)/255), plt.colorbar()
        # plt.imshow(labels[0].squeeze().cpu().permute(1,2,0)/255), plt.colorbar()
        plt.subplot(2,1,2), plt.imshow(labels[0].cpu().squeeze(0)/255), plt.colorbar()
        plt.show()

        break

    print(f"############################")

def create_data_loaders(full_dataset, data_splitting_ratios, seed, batch_size, num_workers):
    """
    Create train, validation, and test dataloaders
    """

    train_set, validation_set, test_set = torch.utils.data.random_split(full_dataset, data_splitting_ratios, torch.Generator().manual_seed(seed))

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, validation_loader, test_loader

def training_loop(model, current_idx, current_data, optimizer, training_performance_dict, loss_fn, cuda_available):
    """
    Trains model using training data

    # TODO
    #                sanity_check(trainloader, model, cuda_available)
    #                save_checkpoint(
    #                    {
    #                        "epoch": run_epoch,
    #                        "state_dict": model.state_dict(),
    #                        "optimizer": optimizer.state_dict(),
    #                    },
    #                    "models/o.pth",
    #                )
    #
    # if j == 300:
    #     break
    # # performance[key].append(average_metric)

    """
    images, labels = current_data
    if cuda_available:
        images = images.cuda()
        labels = labels.cuda()

    optimizer.zero_grad()
    output = model(images)
    print(f"output.size() {output.size()};\
    type(output): {type(output)};\
    pred.type: {output.type()} ")
    # torch.Size([batch_size_, 4, 400, 640]);
    # <class 'torch.Tensor'>;
    # torch.cuda.FloatTensor

    loss = loss_fn(output, labels)
    loss.backward()
    optimizer.step()

    batch_metrics = evaluate(output, labels)

    for key, value in batch_metrics.items():
        # print(f"{key}: {value:.4f}")
        training_performance_dict[key] += value * len(images) # weighted by batch size

    num_samples_processed = len(images)
    current_training_loss = loss.item()

    # Log every X batches
    if current_idx % 50 == 0 or current_idx == 1:
        print(f"Training: Loss at {current_idx} mini-batch {loss.item():.4f}")
    return current_training_loss, num_samples_processed

def validation_loop(model, current_idx, current_data, optimizer, validation_performance_dict, loss_fn, cuda_available):
    """
    Validates model using validation data

    # TODO
    #                sanity_check(trainloader, model, cuda_available)
    #                save_checkpoint(
    #                    {
    #                        "epoch": run_epoch,
    #                        "state_dict": model.state_dict(),
    #                        "optimizer": optimizer.state_dict(),
    #                    },
    #                    "models/o.pth",
    #                )
    #
    # if j == 300:
    #     break
    # # performance[key].append(average_metric)

    """
    images, labels = current_data
    if cuda_available:
        images = images.cuda()
        labels = labels.cuda()

    optimizer.zero_grad()
    output = model(images)
    print(f"output.size() {output.size()};\
    type(output): {type(output)};\
    pred.type: {output.type()} ")
    # torch.Size([batch_size_, 4, 400, 640]);
    # <class 'torch.Tensor'>;
    # torch.cuda.FloatTensor

    loss = loss_fn(output, labels)

    batch_metrics = evaluate(output, labels)

    for key, value in batch_metrics.items():
        # print(f"{key}: {value:.4f}")
        validation_performance_dict[key] += value * len(images) # weighted by batch size

    num_samples_processed = len(images)
    current_validation_loss = loss.item()

    # Log every X batches
    if current_idx % 50 == 0 or current_idx == 1:
        print(f"Validation: Loss at {current_idx} mini-batch {loss.item():.4f}")

    return current_validation_loss, num_samples_processed

def evaluate_model(model, test_loader, device):
    """
    Evaluate model using test data
    """

    total_elements = 0.0
    num_matches = 0
    with torch.set_grad_enabled(False):
        for data in test_loader:
            images, labels = data
            test_output = model(images.to(device))
            _, predicted = torch.max(test_output.data, 1)

            # Compares how many elements in predicted and labels tensors match
            # and counts them all
            num_matches += (predicted == labels.to(device)).sum().item()

            # Total number of elements in label tensor
            total_elements += torch.numel(labels)

    test_accuracy = (num_matches / total_elements) * 100

    return test_accuracy

def performance_file_writer(folder_path, file_prefix, performance_dict, current_time_stamp):
    """
    Writes performance metrics to .json file
    """

    json_file = folder_path + file_prefix + current_time_stamp + ".json"
    text = json.dumps(performance_dict, indent=4)
    with open(json_file, "w") as out_file_obj:
        out_file_obj.write(text)

def loss_values_file_writer(folder_path, file_prefix, loss_values, current_time_stamp):
    """
    Writes loss values to .csv file
    """
    loss_file = folder_path + file_prefix + current_time_stamp + ".csv"
    with open(loss_file, "w") as out_file_obj:
        for loss in loss_values:
            out_file_obj.write(f"{loss}\n")

def test_accuracy_file_writer(folder_path, test_accuracy, current_time_stamp, pretrained_model_flag):
    """
    Writes test accuracy value to .csv file
    """

    accuracy_file = ""
    if pretrained_model_flag:
        accuracy_file = folder_path + "/accuracy_value_for_reevaluation_" + current_time_stamp + ".csv"
    else:
        accuracy_file = folder_path + "/accuracy_value_" + current_time_stamp + ".csv"

    os.makedirs(folder_path, exist_ok=True)

    with open(accuracy_file, "w") as out_file_obj:
        out_file_obj.write(f"{test_accuracy}\n")
