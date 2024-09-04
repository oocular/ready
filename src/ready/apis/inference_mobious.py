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
from src.ready.utils.datasets import MobiousDataset
from src.ready.utils.utils import get_working_directory, set_data_directory

#TODO
#from sklearn.metrics import (
#    jaccard_score, f1_score, recall_score, precision_score, accuracy_score, fbeta_score)

#TODO
# Make sure we have a common path for models to avoid looking where the model path is!

#TODO
#Add argument to put path of data and name of model

if __name__ == "__main__":
    #set_data_directory("datasets/mobious")
    set_data_directory("ready/data/mobious/sample-frames")
    print(os.getcwd())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #trainset = MobiousDataset("MOBIOUS/train") #for  set_data_directory("datasets/mobious/MOBIOUS")
    trainset = MobiousDataset("test640x400") #for set_data_directory("ready/data/mobious/sample-frames")
    print("Length of trainset:", len(trainset))

    batch_size_ = 8  # 8 original
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_, shuffle=True, num_workers=4
    )
    print(f"trainloader.batch_size {trainloader.batch_size}")

    ### PTH model
    #model_name = "_weights_27-08-24_05-23_trained_10epochs_8batch_1143lentrainset"
    #model_name = "_weights_02-09-24_21-02"
    #model_name = "weights_02-09-24_22-24_trained10e_8batch_1143trainset"
    model_name="_weights_03-09-24_19-16"

    #Epoch 100: 
              #Average loss @ epoch: 9.622711725168294
              #Saved PyTorch Model State to weights/_weights_03-09-24_19-16.pth
              #Elapsed time for the training loop: 48.18073609670003 (mins)

    #Epoch 20: loss no-weights
              #Average loss @ epoch: 11.027751895931218
              #Saved PyTorch Model State to weights/_weights_03-09-24_22-34.pth
              #Elapsed time for the training loop: 9.677963574727377 (mins)

    #Epoch 20: loss with weights
              #Average loss @ epoch: 14.233737432039701
              #Saved PyTorch Model State to weights/_weights_03-09-24_22-58.pth
              #Elapsed time for the training loop: 9.664288135369619 (mins)

    checkpoint_path = "weights/"+str(model_name)+".pth"
    model = UNet(nch_in=3, nch_out=4)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    #### ONNX model
    onnx_checkpoint_path = "weights/"+str(model_name)+"-sim.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_checkpoint_path, providers=["CPUExecutionProvider"]) 
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
            image=images[0].unsqueeze(0)
            labels = labels.cuda()
            label=labels[0].unsqueeze(0)
            #print(image.size()) #torch.Size([1, 3, 400, 640])
            #print(images.size()) #torch.Size([5, 3, 400, 640])
            #print(label.size()) #torch.Size([1, 4, 400, 640])
            #print(labels.size()) #torch.Size([5, 4, 400, 640])

        ##PTH model
        outputs = model(image)
        #print(outputs.size()) #torch.Size([1, 4, 400, 640])
        outputs = torch.argmax(outputs[0], 0)
        #print(outputs.size()) #torch.Size([400, 640])
        pred = torch.sigmoid(model(images[0].unsqueeze(0)))
        #print(pred.size(), pred[0].size()) #torch.Size([1, 4, 400, 640]) torch.Size([4, 400, 640])

        ##ONNX model
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(images[0].unsqueeze(0))}
        ort_outs = np.asarray(ort_session.run(None, ort_inputs))
        ort_outs = torch.tensor(ort_outs)
        ort_outs = ort_outs.squeeze(0).squeeze(0)
        #print(ort_outs.size()) #torch.Size([4, 400, 640])
        ort_outs = torch.argmax(ort_outs,0)
        #print(ort_outs.size())#torch.Size([400, 640])

        #plt.figure()
        image = torch.permute(image, (0, 2, 3, 1))
        ax[0].imshow((image[0]).to(torch.long).squeeze(0).cpu())
        ax[1].imshow(label.permute(0,2,3,1).squeeze(0).cpu())
        ax[2].imshow(outputs.squeeze(0).cpu())
        ax[3].imshow(ort_outs.cpu())
        #plt.show()

        if j == 2:
            break

    plt.show()
