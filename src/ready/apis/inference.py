"""
Inference
See skmetrics: https://github.com/MatejVitek/SSBC/blob/master/evaluation/segmentation.py
"""

import os

import onnxruntime
import matplotlib.image as mimg
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.ready.models.unet import UNet
from src.ready.utils.datasets import EyeDataset
from src.ready.utils.utils import get_working_directory, set_data_directory

#TODO
#from sklearn.metrics import (
#    jaccard_score, f1_score, recall_score, precision_score, accuracy_score, fbeta_score)


if __name__ == "__main__":
    set_data_directory("datasets/openEDS")
    print(os.getcwd())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    trainset = EyeDataset("openEDS/openEDS/val3frames")  # train #test #validation #val3frames
    print("Length of trainset:", len(trainset))

    batch_size_ = 8  # 8 original
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_, shuffle=True, num_workers=4
    )
    print(f"trainloader.batch_size {trainloader.batch_size}")


    ### PTH model
    checkpoint_path = "weights/trained_models_in_cricket/model-5jul2024.pth"
    model = UNet(nch_in=1, nch_out=4)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    ### ONNX model
    ort_session = onnxruntime.InferenceSession("weights/trained_models_in_cricket/model-5jul2024.onnx", providers=["CPUExecutionProvider"]) 
    #UserWarning: Specified provider 'CUDAExecutionProvider' is not in available
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    ### MAIN LOOP
    f, ax = plt.subplots(1, 4)
    cuda_available = torch.cuda.is_available()
    for j, data in enumerate(trainloader, 1):
        print(j)
        images, labels = data
        if cuda_available:
            images = images.cuda()
            labels = labels.cuda()

        #print(images[0].unsqueeze(0).size()) #torch.Size([1, 1, 400, 640])
        #print(labels[0].unsqueeze(0).size()) #torch.Size([1, 400, 640])


        ##PTH model
        outputs = model(images[0].unsqueeze(0))  # torch.Size([1, 4, 400, 640])
        outputs = torch.argmax(outputs[0], 0)  # torch.Size([400, 640])
        #print(outputs.size())
        pred = torch.sigmoid(model(images[0].unsqueeze(0)))
        #print(pred.size(), pred[0].size()) #torch.Size([1, 4, 400, 640]) torch.Size([4, 400, 640])


	##ONNX model
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(images[0].unsqueeze(0))}
        ort_outs = np.asarray(ort_session.run(None, ort_inputs))
        ort_outs = torch.tensor(ort_outs)
        ort_outs = ort_outs.squeeze(0).squeeze(0) #torch.Size([4, 400, 640])
        ort_outs = torch.argmax(ort_outs,0) #torch.Size([400, 640])


        # plt.figure()
        ax[0].imshow((images[0] * 255).to(torch.long).squeeze(0).cpu())
        ax[1].imshow(labels[0].squeeze(0).cpu())
        ax[2].imshow(outputs.squeeze(0).cpu())
        ax[3].imshow(ort_outs.cpu())
        plt.show()

        if j == 2:
            break





