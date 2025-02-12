import glob
import os
from pathlib import Path
from random import randrange
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset

from ready.models.unetvit import UNetViT
from ready.utils.datasets import MOBIOUSDataset_unetvit
from ready.utils.utils import (DATASET_PATH, MODELS_PATH, DeviceDataLoader,
                               get_default_device, precision, recall)


with open(str(Path().absolute())+"/tests/config_test.yml", "r") as file:
    config_yaml = yaml.load(file, Loader=yaml.FullLoader)

def test_MOBIOUSDataset_unetvit():
    """
    Test MOBIOUSDataset_unetvit class
    pytest -vs tests/test_unetvit.py::test_MOBIOUSDataset_unetvit
        TODO:
            - Use 640x400 image size
            - Test mask to show sclera, iris, pupil and background
    """
    # Define transforms - note we do ToTensor in the dataset class
    color_shift = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
    blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))
    resize = transforms.Resize((512, 512))  # Standard size for UNet-ViT
    t = transforms.Compose([resize, color_shift, blurriness])

    DATASET_PATH = os.path.join(str(Path.home()), config_yaml['ABS_DATA_PATH'])

    # Use the new dataset class
    dataset = MOBIOUSDataset_unetvit(DATASET_PATH, training=True, transform=t)
    single_set = dataset[ randrange(len(dataset)) ]

    # Print detailed shape information
    logger.info(f"")
    logger.info(f"len(dataset) : {len(dataset)}")
    logger.info(f"single_set[0].shape (IMAGE): {single_set[0].shape}")
    logger.info(f"single_set[1].shape (MASK): {single_set[1].shape}")
    logger.info(f"Mask unique values: {torch.unique(single_set[1])}")
    logger.info(f"Mask dtype: {single_set[1].dtype}")

    assert single_set[0].shape == (
        3,
        512,
        512,
    ), f"Expected image shape (3, 512, 512), but got {single_set[0].shape}"
    assert single_set[1].shape == (
        512,
        512,
    ), f"Expected mask shape (512, 512), but got {single_set[1].shape}"

    # plot image and mask
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.imshow(np.moveaxis(single_set[0].numpy(), 0, -1))
    plt.title("image")
    plt.subplot(1, 2, 2)
    plt.imshow(single_set[1].numpy())
    plt.title("mask")
    plt.show()



def test_train_pipeline():
    """
    Test train pipeline with MOBIOUS dataset for UNetViT
    pytest -vs tests/test_unetvit.py::test_train_pipeline
    """
    DATASET_PATH = os.path.join(str(Path.home()), config_yaml['ABS_DATA_PATH'])
    MODELS_PATH = DATASET_PATH + "/models" #TODO add an absolute model path

    device = get_default_device()

    # Data preparation
    color_shift = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
    blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))
    resize = transforms.Resize((512, 512))
    t = transforms.Compose([resize, color_shift, blurriness])

    dataset = MOBIOUSDataset_unetvit(DATASET_PATH, training=True, transform=t)
    test_num = int(0.1 * len(dataset))

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [len(dataset) - test_num, test_num],
        generator=torch.Generator().manual_seed(101),
    )

    BATCH_SIZE = 4
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    train_dataloader = DeviceDataLoader(train_dataloader, device)
    test_dataloader = DeviceDataLoader(test_dataloader, device)

    logger.info(f"Number of test batches: {len(test_dataloader)}")
    logger.info(f"Number of train batches: {len(train_dataloader)}")

    assert len(test_dataloader) > 0
    assert len(train_dataloader) > 0

    # Model initialization
    model = UNetViT(n_channels=3, n_classes=6, bilinear=True)
    model = model.to(device)

    # Training parameters
    NUM_EPOCHS = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        # Training phase
        for batch_idx, (images, masks) in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, masks.long())
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch: {epoch+1}/{NUM_EPOCHS}, Batch: {batch_idx}/{len(train_dataloader)}, Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in test_dataloader:
                outputs = model(images)
                loss = criterion(outputs, masks.long())
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_dataloader)
        
        logger.info(f'Epoch {epoch+1} - Train Loss: {avg_train_loss:.5f}, Val Loss: {avg_val_loss:.5f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 
                      f"{MODELS_PATH}/unetvit_epochs_{epoch}_valloss_{avg_val_loss:.5f}.pth")
            logger.info(f"Saved new best model with validation loss: {avg_val_loss:.5f}")

    # Calculate final metrics
    model.eval()
    final_val_loss = 0.0
    precision_list = []
    recall_list = []
    
    with torch.no_grad():
        for images, masks in test_dataloader:
            outputs = model(images)
            loss = criterion(outputs, masks.long())
            final_val_loss += loss.item()
            
            # Calculate precision and recall
            precision_list.append(precision(masks, outputs))
            recall_list.append(recall(masks, outputs))
    
    final_val_loss /= len(test_dataloader)
    final_precision = np.nanmean(precision_list, axis=0)
    final_recall = np.nanmean(recall_list, axis=0)
    
    # Calculate F1 score (excluding background class)
    f1_score = (
        2 * (sum(final_precision[:-1]) / 5 * sum(final_recall[:-1]) / 5) /
        (sum(final_precision[:-1]) / 5 + sum(final_recall[:-1]) / 5)
    )
    
    logger.info(f"\nFinal Results:")
    logger.info(f"Validation Loss: {final_val_loss:.5f}")
    logger.info(f"Precision: {final_precision}")
    logger.info(f"Recall: {final_recall}")
    logger.info(f"F1 Score: {f1_score:.5f}")

    # Test assertions
    assert final_val_loss < float('inf'), "Training failed to converge"
    assert not np.isnan(f1_score), "F1 score is NaN"
    assert f1_score > 0, "F1 score should be positive"

def test_quick_train_pipeline():
    """
    Quick training pipeline that runs just one epoch to generate a model file for testing
    pytest -vs tests/test_unetvit.py::test_quick_train_pipeline
    Uses a smaller subset of the data (20 samples)
    Uses a smaller batch size (2)
    Runs for just one epoch
    Saves the model with a distinctive name
    """
    DATASET_PATH = os.path.join(str(Path.home()), config_yaml['ABS_DATA_PATH'])
    MODELS_PATH = DATASET_PATH + "/models" #TODO add an absolute model path
    
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_PATH, exist_ok=True)

    device = get_default_device()

    # Data preparation - using minimal data
    color_shift = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
    blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))
    resize = transforms.Resize((512, 512))
    t = transforms.Compose([resize, color_shift, blurriness])

    dataset = MOBIOUSDataset_unetvit(DATASET_PATH, training=True, transform=t)
    
    # Use a smaller subset of data for quick testing
    subset_size = min(len(dataset), 20)  # Use only 20 samples or less if dataset is smaller
    indices = torch.randperm(len(dataset))[:subset_size]
    dataset = torch.utils.data.Subset(dataset, indices)
    
    test_num = int(0.2 * len(dataset))  # 20% for testing

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [len(dataset) - test_num, test_num],
        generator=torch.Generator().manual_seed(101),
    )

    BATCH_SIZE = 2  # Smaller batch size for quicker training
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    train_dataloader = DeviceDataLoader(train_dataloader, device)
    test_dataloader = DeviceDataLoader(test_dataloader, device)

    # Model initialization
    model = UNetViT(n_channels=3, n_classes=6, bilinear=True)
    model = model.to(device)

    # Training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Single epoch training
    model.train()
    train_loss = 0.0
    
    # Training phase
    for batch_idx, (images, masks) in enumerate(train_dataloader):
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, masks.long())
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        logger.info(f'Batch: {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}')
    
    avg_train_loss = train_loss / len(train_dataloader)
    
    # Quick validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in test_dataloader:
            outputs = model(images)
            loss = criterion(outputs, masks.long())
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(test_dataloader)
    
    # Save model
    save_path = f"{MODELS_PATH}/unetvit_quick_test_valloss_{avg_val_loss:.5f}.pth"
    torch.save(model.state_dict(), save_path)
    
    logger.info(f"\nQuick Training Results:")
    logger.info(f"Train Loss: {avg_train_loss:.5f}")
    logger.info(f"Validation Loss: {avg_val_loss:.5f}")
    logger.info(f"Model saved to: {save_path}")

    return save_path  # Return the path to the saved model



def test_inference():
    """
    Test inference with MOBIUS dataset with detailed visualization
    pytest -vs tests/test_unetvit.py::test_inference
    """
    CURRENT_PWD = Path().absolute()
    DATASET_PATH = str(CURRENT_PWD) + "/data/test-samples/MOBIUS"
    MODELS_PATH = DATASET_PATH + "/models"
    device = get_default_device()

    # Data preparation
    color_shift = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
    blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))
    resize = transforms.Resize((512, 512))
    t = transforms.Compose([resize, color_shift, blurriness])

    dataset = MOBIOUSDataset_unetvit(DATASET_PATH, training=True, transform=t)
    test_num = int(0.1 * len(dataset))

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [len(dataset) - test_num, test_num],
        generator=torch.Generator().manual_seed(101),
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    test_dataloader = DeviceDataLoader(test_dataloader, device)

    # Load PyTorch model
    input_model_name = "unetvit_quick_test_valloss_7.23219.pth"
    model_name = input_model_name[:-4]
    model = UNetViT(n_channels=3, n_classes=6, bilinear=True).to(device)
    model.load_state_dict(torch.load(MODELS_PATH + "/" + input_model_name))
    model.eval()

    # ONNX setup
    onnx_path = MODELS_PATH + "/" + str(model_name) + "-sim.onnx"
    if not os.path.exists(onnx_path):
        dummy_input = torch.randn(1, 3, 512, 512, device=device)
        torch.onnx.export(model, dummy_input, onnx_path,
                         export_params=True,
                         opset_version=11,
                         do_constant_folding=True,
                         input_names=['input'],
                         output_names=['output'],
                         dynamic_axes={'input': {0: 'batch_size'},
                                     'output': {0: 'batch_size'}})
        logger.info(f"ONNX model saved to: {onnx_path}")

    ort_session = onnxruntime.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    for i, (image, label) in enumerate(test_dataloader):
        with torch.no_grad():
            # Get the original mask path
            image_idx = test_dataset.indices[i]  # Get the original index from the subset
            img_path = dataset.img_paths[image_idx]
            mask_path = dataset.mask_paths[image_idx]  # Use the mask_paths directly from dataset

            # Verify paths
            logger.info(f"Processing image: {img_path}")
            logger.info(f"Corresponding mask: {mask_path}")

            # Load and process the original mask
            original_mask = Image.open(mask_path)
            original_mask = resize(original_mask)

            # Model predictions
            outputs = model(image)
            pred_softmax = F.softmax(outputs, dim=1)
            pred_argmax_softmax = torch.argmax(pred_softmax, dim=1)

            # ONNX predictions
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image)}
            ort_outs = ort_session.run(None, ort_inputs)[0]
            ort_outs = torch.from_numpy(ort_outs)
            if len(ort_outs.shape) == 3:
                ort_outs = ort_outs.unsqueeze(0)
            ort_outs_argmax = torch.argmax(ort_outs[0], dim=0)

            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Eye Region Segmentation Analysis', fontsize=16)

            # Input Image
            input_img = image[0].permute(1, 2, 0).cpu().numpy()
            input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
            axes[0, 0].imshow(input_img)
            axes[0, 0].set_title('Input Image')
            axes[0, 0].axis('off')

            # Ground Truth Mask
            axes[0, 1].imshow(np.array(original_mask))
            axes[0, 1].set_title('Ground Truth')
            axes[0, 1].axis('off')

            # PyTorch Prediction
            axes[0, 2].imshow(pred_argmax_softmax[0].cpu().numpy())
            axes[0, 2].set_title('PyTorch Prediction')
            axes[0, 2].axis('off')

            # ONNX Prediction
            axes[1, 0].imshow(ort_outs_argmax.cpu().numpy())
            axes[1, 0].set_title('ONNX Prediction')
            axes[1, 0].axis('off')

            # Probability map
            axes[1, 1].imshow(pred_softmax[0, 1].cpu().numpy(), cmap='hot')
            axes[1, 1].set_title('Sclera Probability')
            axes[1, 1].axis('off')
            plt.colorbar(plt.cm.ScalarMappable(cmap='hot'), ax=axes[1, 1])

            # Legend
            axes[1, 2].axis('off')
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, fc='k', label='Background'),
                plt.Rectangle((0, 0), 1, 1, fc='r', label='Sclera'),
                plt.Rectangle((0, 0), 1, 1, fc='g', label='Iris'),
                plt.Rectangle((0, 0), 1, 1, fc='b', label='Pupil')
            ]
            axes[1, 2].legend(handles=legend_elements, loc='center', title='Regions')

            plt.tight_layout()
            plt.show()

            if i == 2:  # Show only first 3 images
                break