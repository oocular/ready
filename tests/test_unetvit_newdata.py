import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import torch
import torchvision.transforms as transforms
from loguru import logger
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import glob

from ready.models.unetvit import UNetViT
from ready.utils.utils import (
    DATASET_PATH,
    MODELS_PATH,
    DeviceDataLoader,
    get_default_device,
    precision,
    recall,
)


class MOBIUSDataset(Dataset):
    """Dataset class for MOBIUS data with numbered folders."""
    
    def __init__(self, data_dir, training=True, transform=None):
        """
        Args:
            data_dir (str): Root directory of MOBIUS dataset
            training (bool): Whether this is training data
            transform (callable, optional): Transform to be applied to images
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.training = training
        
        # Get all numbered folders
        self.folder_numbers = []
        for i in range(1, 36):  # 1 to 35
            if (self.data_dir / "Images" / str(i)).exists():
                self.folder_numbers.append(str(i))
        
        # Collect all image paths and corresponding mask paths
        self.img_paths = []
        self.mask_paths = []
        
        for folder_num in self.folder_numbers:
            img_folder = self.data_dir / "Images" / folder_num
            mask_folder = self.data_dir / "Masks" / folder_num
            
            # Get all image files in this folder
            img_files = sorted(glob.glob(str(img_folder / "*")))
            mask_files = sorted(glob.glob(str(mask_folder / "*")))
            
            # Verify matching pairs
            for img_path, mask_path in zip(img_files, mask_files):
                img_name = Path(img_path).stem
                mask_name = Path(mask_path).stem
                if img_name == mask_name:  # Only add if names match
                    self.img_paths.append(img_path)
                    self.mask_paths.append(mask_path)
        
        if len(self.img_paths) == 0:
            raise RuntimeError(f"No images found in {data_dir}")
            
        print(f"Found {len(self.img_paths)} images in {len(self.folder_numbers)} folders")
    def __len__(self):
        return len(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load and process image
        image = Image.open(img_path).convert("RGB")
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        
        # Load and process mask - should be binary mask with values 0 and 1
        mask = Image.open(mask_path)
        mask = np.array(mask)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]  # Take first channel
        # Convert from 0/255 to 0/1
        mask = (mask > 0).astype(np.int64)  # Convert to long for PyTorch
        mask = torch.from_numpy(mask)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            # Make sure mask is also resized to match image size if resize is part of transforms
            if any(isinstance(t, transforms.Resize) for t in self.transform.transforms):
                resize = next(t for t in self.transform.transforms if isinstance(t, transforms.Resize))
                mask = transforms.Resize(resize.size, interpolation=transforms.InterpolationMode.NEAREST)(mask.unsqueeze(0)).squeeze(0)
        
        return image, mask


def test_segDataset():
    """
    Test segDataset class for MOBIUS dataset
    pytest -vs tests/test_unetvit_newdata.py::test_segDataset
    """
    # Define transforms - note we do ToTensor in the dataset class
    color_shift = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
    blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))
    resize = transforms.Resize((512, 512))  # Standard size for UNet-ViT
    t = transforms.Compose([resize, color_shift, blurriness])
    CURRENT_PWD = Path().absolute()
    DATASET_PATH = str(CURRENT_PWD) + "/data/test-samples/MOBIUS"
    # Use the new dataset class
    dataset = MOBIUSDataset(DATASET_PATH, training=True, transform=t)
    single_set = dataset[0]

    CURRENT_PWD = Path().absolute()
    DATASET_PATH = str(CURRENT_PWD) + "/data/test-samples/MOBIUS"

    # Use the new dataset class
    dataset = MOBIUSDataset(DATASET_PATH, training=True, transform=t)
    single_set = dataset[0]

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


def test_inference():
    """
    Test inference with MOBIUS dataset
    pytest -vs tests/test_unetvit_newdata.py::test_inference
    """
    CURRENT_PWD = Path().absolute()
    DATASET_PATH = str(CURRENT_PWD) + "/data/test-samples/MOBIUS"
    MODELS_PATH = DATASET_PATH + "/models"
    device = get_default_device()

    device = get_default_device()

    color_shift = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
    blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))
    resize = transforms.Resize((512, 512))
    t = transforms.Compose([resize, color_shift, blurriness])
    
    dataset = MOBIUSDataset(DATASET_PATH, training=True, transform=t)

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

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    train_dataloader = DeviceDataLoader(train_dataloader, device)
    test_dataloader = DeviceDataLoader(test_dataloader, device)

    logger.info(f"Number of test batches: {len(test_dataloader)}")
    logger.info(f"Number of train batches: {len(train_dataloader)}")

    input_model_name = "unetvit_epochs_0_valloss_2.07737.pth"
    model_name = input_model_name[:-4]
    model = UNetViT(n_channels=3, n_classes=6, bilinear=True).to(device)
    model.load_state_dict(torch.load(MODELS_PATH + "/" + input_model_name))
    model.eval()

    ## ONNX model
    onnx_checkpoint_path = MODELS_PATH + "/" + str(model_name) + "-sim.onnx"
    ort_session = onnxruntime.InferenceSession(
        onnx_checkpoint_path, providers=["CPUExecutionProvider"]
    )

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )
    for i, (image_batch, ground_truth_masks) in enumerate(test_dataloader):
        for batch_j in range(len(image_batch)):
            image_batch_j = image_batch[batch_j : batch_j + 1]

    for i, (image_batch, ground_truth_masks) in enumerate(test_dataloader):
        for batch_j in range(len(image_batch)):
            image_batch_j = image_batch[batch_j : batch_j + 1]

            # Pytorch model inference
            result = model(image_batch_j)
            mask = torch.argmax(result, axis=1).cpu().detach().numpy()[0]
            im = (
                np.moveaxis(image_batch[batch_j].cpu().detach().numpy(), 0, -1).copy()
                * 255
            )
            im = im.astype(int)
            gt_mask = ground_truth_masks[batch_j].cpu()

            # onnx model inference
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image_batch_j)}
            ort_outs = torch.tensor(np.asarray(ort_session.run(None, ort_inputs)))
            ort_outs = ort_outs.squeeze(0).squeeze(0)
            ort_outs_argmax = torch.argmax(ort_outs, dim=0).cpu().detach().numpy()
            plt.figure(figsize=(12, 12))
            plt.subplot(1, 4, 1)
            plt.imshow(im)
            plt.title("image")
            plt.subplot(1, 4, 2)
            plt.imshow(gt_mask)
            plt.title("Ground Truth Mask")
            plt.subplot(1, 4, 3)
            plt.imshow(mask)
            plt.title("Pytorch predicted mask")
            plt.subplot(1, 4, 4)
            plt.imshow(ort_outs_argmax)
            plt.title("Onnx predicted mask")
            plt.show()

            plt.figure(figsize=(12, 12))

            plt.subplot(1, 4, 1)
            plt.imshow(im)
            plt.title("image")

            plt.subplot(1, 4, 2)
            plt.imshow(gt_mask)
            plt.title("Ground Truth Mask")

            plt.subplot(1, 4, 3)
            plt.imshow(mask)
            plt.title("Pytorch predicted mask")

            plt.subplot(1, 4, 4)
            plt.imshow(ort_outs_argmax)
            plt.title("Onnx predicted mask")

            plt.show()

    pred_list = []
    gt_list = []
    precision_list = []
    recall_list = []
    for i, (image_batch, ground_truth_masks) in enumerate(test_dataloader):
        for batch_j in range(len(image_batch)):
            result = model(image_batch.to(device)[batch_j : batch_j + 1])
            precision_list.append(precision(ground_truth_masks[batch_j], result))
            recall_list.append(recall(ground_truth_masks[batch_j], result))

    final_precision = np.nanmean(precision_list, axis=0)
    final_recall = np.nanmean(recall_list, axis=0)
    f1_score = (
        2
        * (sum(final_precision[:-1]) / 5 * sum(final_recall) / 5)
        / (sum(final_precision[:-1]) / 5 + sum(final_recall) / 5)
    )

    logger.info(f"nanmean of precision_list : {final_precision}")
    logger.info(f"nanmean of recall_list : {final_recall}")
    logger.info(f"Final precision : {sum(final_precision[:-1])/5}")
    logger.info(f"Final recall : {sum(final_recall)/5}")
    logger.info(f"Final f1_score : {f1_score}")