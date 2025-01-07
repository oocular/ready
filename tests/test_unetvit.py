import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import torch
import torchvision.transforms as transforms
from loguru import logger
from pathlib import Path

from ready.models.unetvit import UNetViT
from ready.utils.datasets import segDataset
from ready.utils.utils import (
    DATASET_PATH,
    MODELS_PATH,
    DeviceDataLoader,
    get_default_device,
    precision,
    recall,
)


def test_segDataset():
    """
    Test segDataset class
    pytest -vs tests/test_unetvit.py::test_segDataset

    References
    https://www.youtube.com/watch?v=jvZm8REF2KY

    """
    color_shift = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
    blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))
    t = transforms.Compose([color_shift, blurriness])

    CURRENT_PWD = Path().absolute()
    DATASET_PATH = str(CURRENT_PWD) + "/data/test-samples/semantic-segmentation-aerial-imagery"

    dataset = segDataset(DATASET_PATH, training=True, transform=t)
    single_set = dataset[0]

    logger.info(f"")
    logger.info(f"len(dataset) : {len(dataset)}")
    logger.info(f"len(single_set) : {len(single_set)}")
    logger.info(f"single_set[0].shape (IMAGE): {single_set[0].shape}")
    logger.info(f"single_set[1].shape (MASK) : {single_set[1].shape}")

    assert single_set[0].shape == (
        3,
        512,
        512,
    ), f"Expected image shape (3, 512, 512), but got {d[0].shape}"
    assert single_set[1].shape == (
        512,
        512,
    ), f"Expected mask shape (512, 512), but got {d[1].shape}"

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
    Test inference
    pytest -vs tests/test_unetvit.py::test_inference
    """
    CURRENT_PWD = Path().absolute()
    DATASET_PATH = str(CURRENT_PWD) + "/data/test-samples/semantic-segmentation-aerial-imagery"
    MODELS_PATH = DATASET_PATH + "/models"

    device = get_default_device()

    color_shift = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
    blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))
    t = transforms.Compose([color_shift, blurriness])
    dataset = segDataset(DATASET_PATH, training=True, transform=t)

    test_num = int(0.1 * len(dataset))
    # print(f'test data : {test_num}')#7

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [len(dataset) - test_num, test_num],
        generator=torch.Generator().manual_seed(101),
    )

    BACH_SIZE = 4
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BACH_SIZE, shuffle=True, num_workers=0
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BACH_SIZE, shuffle=False, num_workers=0
    )

    train_dataloader = DeviceDataLoader(train_dataloader, device)
    test_dataloader = DeviceDataLoader(test_dataloader, device)

    assert (
        len(test_dataloader) == 1
    ), f"Expected 1 batches, but got {len(test_dataloader)}"
    assert (
        len(train_dataloader) == 5
    ), f"Expected 5 batches, but got {len(train_dataloader)}"

    #TODO: Add argument to select model (https://stackoverflow.com/questions/40880259/)
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

    # UserWarning: Specified provider 'CUDAExecutionProvider' is not in available
    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    # TODO: Add condition to plot this section
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
            im = (
                np.moveaxis(image_batch[batch_j].cpu().detach().numpy(), 0, -1).copy()
                * 255
            )
            im = im.astype(int)
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
