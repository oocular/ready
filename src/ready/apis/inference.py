"""
Inference
See skmetrics: https://github.com/MatejVitek/SSBC/blob/master/evaluation/segmentation.py
See pixel_accuracy, mIoU : https://github.com/tanishqgautam/Drone-Image-Semantic-Segmentation/blob/main/semantic-segmentation-pytorch.ipynb
"""

import os

import onnxruntime
import matplotlib.image as mimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.ready.models.unet import UNet
from src.ready.utils.datasets import EyeDataset
from src.ready.utils.utils import get_working_directory, set_data_directory

#TODO
#from sklearn.metrics import (
#    jaccard_score, f1_score, recall_score, precision_score, accuracy_score, fbeta_score)


if __name__ == "__main__":
    # set_data_directory("datasets/openEDS")
    set_data_directory("ready/data/openEDS")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # trainset = EyeDataset("openEDS/openEDS/validation")  # train #test #validation (Length of trainset: 2384) 
    trainset = EyeDataset("sample-frames/val3frames") #for     set_data_directory("ready/data/openEDS")

    print("Length of trainset:", len(trainset))

    batch_size_ = 8 # 8 original
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_, shuffle=True, num_workers=4
    )
    print(f"trainloader.batch_size {trainloader.batch_size}")


    ### PTH model
    # model_name="model3ch-23jul2024t0716"
    model_name="_weights_10-09-24_23-53-45"
            #run_epoch = 100
            #Average loss @ epoch: 0.0025765099562704563
            #Saved PyTorch Model State to models/_weights_10-09-24_23-53-45.pth
            #Elapsed time for the training loop: 1.3849741021792095 (mins)

    checkpoint_path = "models/"+str(model_name)+".pth"

    # model = UNet(nch_in=1, nch_out=4)
    model = UNet(nch_in=3, nch_out=4)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    ### ONNX model
    onnx_checkpoint_path = "models/"+str(model_name)+"-sim.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_checkpoint_path, providers=["CPUExecutionProvider"])
    #UserWarning: Specified provider 'CUDAExecutionProvider' is not in available
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    # MAIN LOOP
    f, ax = plt.subplots(7, 6)
    cuda_available = torch.cuda.is_available()
    for j, data in enumerate(trainloader, 1):
        print(j)
        images, labels = data
        if cuda_available:
            images = images.cuda()
            image=images[0].unsqueeze(0)
            labels = labels.cuda()
            label=labels[0].unsqueeze(0)
            ## images
            # print(f"images.size() {images.size()}") #torch.Size([batch_size_, 3, 400, 640])
            # print(f"image.size() {image.size()}") #torch.Size([1, 3, 400, 640])
            # print(f"images.size() {images.size()}; type(labels): {type(images)}; pred.type: {images.type()} ")
            # torch.Size([3, 3, 400, 640]); <class 'torch.Tensor'>; torch.cuda.FloatTensor 
            ## labels
            # print(f"labels.size() {labels.size()}") #torch.Size([batch_size_, 400, 640])
            # print(f"label.size() {label.size()}") #torch.Size([1, 400, 640])
            # print(f"labels.size() {labels.size()}; type(labels): {type(labels)}; pred.type: {labels.type()} ")
            # torch.Size([batch_size_, 400, 640]),  <class 'torch.Tensor'>, torch.cuda.LongTensor 

        ##PTH model
        outputs = model(image)
        # print(f"outputs.size() {outputs.size()}") #outputs.size() torch.Size([1, 4, 400, 640])
        # print(outputs[0].size())          #torch.Size([4, 400, 640])
        # print(outputs.squeeze(0).size())  #torch.Size([4, 400, 640])

        ##PREDICTION
        pred_softmax = F.softmax( outputs, dim=1 ) 
        # print(f"pred.size() {pred.size()}, type(pred): {type(pred)} pred.type: {pred.type()} ")
        # pred.size() torch.Size([1, 4, 400, 640]), type(pred): <class 'torch.Tensor'> pred.type: torch.cuda.FloatTensor 


        ## PREDICTION argmax(softmax(x))
        pred_argmax_softmax = torch.argmax(F.softmax(outputs, dim=1), dim=1) 
        # print(pred_mask.size()) #torch.Size([1, 400, 640])
        # print(pred_mask.squeeze(0).size()) #torch.Size([400, 640])

        ## PRECTION argmax(x)
        outputs_argmax = torch.argmax(outputs[0], dim=0) 
        # print(outputs_argmax.size()) #torch.Size([400, 640])

	    ##ONNX model
        # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(images[0].unsqueeze(0))}
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image)}
        ort_outs = torch.tensor(  np.asarray(ort_session.run(None, ort_inputs))  ) 
        # print(ort_outs.size()) #torch.Size([1, 1, 4, 400, 640])
        ort_outs = ort_outs.squeeze(0).squeeze(0)
        # print(ort_outs.size()) #torch.Size([4, 400, 640])
        ort_outs_argmax = torch.argmax(ort_outs, dim=0) 
        # print(ort_outs.size()) #torch.Size([400, 640])

        #######################################
        #RAW IMAGE
        ax[0,0].imshow((image.permute(0, 2, 3, 1) * 255).to(torch.long).squeeze(0).cpu())
        ax[0,1].imshow((image.permute(0, 2, 3, 1) * 255)[:,:,:,0].to(torch.long).squeeze(0).cpu())
        ax[0,2].imshow((image.permute(0, 2, 3, 1) * 255)[:,:,:,1].to(torch.long).squeeze(0).cpu())
        ax[0,3].imshow((image.permute(0, 2, 3, 1) * 255)[:,:,:,2].to(torch.long).squeeze(0).cpu())
        ax[0,0].set_ylabel('RAW [400,640,3]')
        ax[0,1].set_title('ch0')
        ax[0,2].set_title('ch1')
        ax[0,3].set_title('ch2')

        #MASKS
        ax[1,0].imshow(label.squeeze(0).cpu())
        ax[1,1].imshow(label.squeeze(0).cpu()>0)
        ax[1,2].imshow(label.squeeze(0).cpu()>1)
        ax[1,3].imshow(label.squeeze(0).cpu()>2)
        ax[1,4].imshow(label.squeeze(0).cpu()>3)
        ax[1,0].set_ylabel('label[400,640]')
        ax[1,1].set_title('label>0')
        ax[1,2].set_title('label>1')
        ax[1,3].set_title('label>2')
        ax[1,4].set_title('label>3')

        ##PREDICITONS SOFTMAX
        ax[2,0].imshow(pred_softmax.permute(0, 2, 3, 1).squeeze(0).detach().cpu())
        ax[2,1].imshow(pred_softmax[:,0,:,:].squeeze(0).detach().cpu())
        ax[2,2].imshow(pred_softmax[:,1,:,:].squeeze(0).detach().cpu())
        ax[2,3].imshow(pred_softmax[:,2,:,:].squeeze(0).detach().cpu())
        ax[2,4].imshow(pred_softmax[:,3,:,:].squeeze(0).detach().cpu())
        ax[2,0].set_title('pred_softmax [400,640,4]')
        ax[2,1].set_title('ch0')
        ax[2,2].set_title('ch1')
        ax[2,3].set_title('ch2')
        ax[2,4].set_title('ch3')

        ##PREDICTIONS
        ax[3,0].imshow(pred_argmax_softmax.squeeze(0).cpu())
        ax[3,1].imshow(pred_argmax_softmax.squeeze(0).cpu()>0)
        ax[3,2].imshow(pred_argmax_softmax.squeeze(0).cpu()>1)
        ax[3,3].imshow(pred_argmax_softmax.squeeze(0).cpu()>2)
        ax[3,4].imshow(pred_argmax_softmax.squeeze(0).cpu()>3)
        ax[3,0].set_title('argmax(softmax(model(image))) [400, 640]')
        ax[3,1].set_title('p_a_s>0')
        ax[3,2].set_title('p_a_s>1')
        ax[3,3].set_title('p_a_s>2')
        ax[3,4].set_title('p_a_s>3')

        #PREDICTIONS argmax()
        ax[4,0].imshow(outputs_argmax.cpu())
        ax[4,1].imshow(outputs_argmax.cpu()>0)
        ax[4,2].imshow(outputs_argmax.cpu()>1)
        ax[4,3].imshow(outputs_argmax.cpu()>2)
        ax[4,4].imshow(outputs_argmax.cpu()>3)
        ax[4,0].set_title('argmax(outputs[0]) [400, 640]')
        ax[4,1].set_title('outputs_argmax>0')
        ax[4,2].set_title('outputs_argmax>1')
        ax[4,3].set_title('outputs_argmax>2')
        ax[4,4].set_title('outputs_argmax>3')

        ax[5,0].imshow(ort_outs_argmax.cpu())
        ax[5,1].imshow(ort_outs_argmax.cpu()>0)
        ax[5,2].imshow(ort_outs_argmax.cpu()>1)
        ax[5,3].imshow(ort_outs_argmax.cpu()>2)
        ax[5,4].imshow(ort_outs_argmax.cpu()>3)
        ax[5,0].set_title('ort_outs_argmax')
        ax[5,1].set_title('ort_outs_argmax>0')
        ax[5,2].set_title('ort_outs_argmax>1')
        ax[5,3].set_title('ort_outs_argmax>2')
        ax[5,4].set_title('ort_outs_argmax>3')

        ax[6,0].imshow(ort_outs.permute(1, 2, 0).cpu())
        ax[6,1].imshow(ort_outs.permute(1, 2, 0)[:,:,0].cpu())
        ax[6,2].imshow(ort_outs.permute(1, 2, 0)[:,:,1].cpu())
        ax[6,3].imshow(ort_outs.permute(1, 2, 0)[:,:,2].cpu())
        ax[6,4].imshow(ort_outs.permute(1, 2, 0)[:,:,3].cpu())
        ax[6,0].set_title('onnx [400,640,4]')
        ax[6,1].set_title('ort_outs[:,:,0]')
        ax[6,2].set_title('ort_outs[:,:,1]')
        ax[6,3].set_title('ort_outs[:,:,2]')
        ax[6,4].set_title('ort_outs[:,:,3]')


        if j == 2:
            break

    plt.show()
