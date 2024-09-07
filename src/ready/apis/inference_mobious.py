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
import torch.nn.functional as F


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
    set_data_directory("ready/data/mobious")
    print(os.getcwd())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #trainset = MobiousDataset("MOBIOUS/train") #for  set_data_directory("datasets/mobious/MOBIOUS")
    trainset = MobiousDataset("sample-frames/test640x400") #for set_data_directory("ready/data/mobious/sample-frames")
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
    #model_name="_weights_03-09-24_19-16"
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

    model_name="_weights_04-09-24_16-31"
        #Epoch 200:
        #Average loss @ epoch: 9.453074308542105
        #Saved PyTorch Model State to weights/_weights_04-09-24_16-31.pth
        #Elapsed time for the training loop: 96.35676774978637 (mins)

    checkpoint_path = "models/"+str(model_name)+".pth"
    model = UNet(nch_in=3, nch_out=4)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    #### ONNX model
    onnx_checkpoint_path = "models/"+str(model_name)+"-sim.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_checkpoint_path, providers=["CPUExecutionProvider"]) 
    #UserWarning: Specified provider 'CUDAExecutionProvider' is not in available
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ### MAIN LOOP
    f, ax = plt.subplots(5, 6)
    cuda_available = torch.cuda.is_available()
    for j, data in enumerate(trainloader, 1):
        print(j)
        images, labels = data
        if cuda_available:
            images = images.cuda()
            image=images[0].unsqueeze(0)
            labels = labels.cuda()
            label=labels[0].unsqueeze(0)
            # print(f"images.size() {images.size()}") #torch.Size([5, 3, 400, 640])
            # print(f"image.size() {image.size()}") #torch.Size([1, 3, 400, 640])
            # print(f"labels.size() {labels.size()}") #torch.Size([5, 4, 400, 640])
            # print(f"label.size() {label.size()}") #torch.Size([1, 4, 400, 640])

        ##PTH model
        outputs = model(image) 
        # print(f"outputs.size() {outputs.size()}") #outputs.size() torch.Size([1, 4, 400, 640])
        # print(outputs[0].size()) #torch.Size([4, 400, 640])
        # print(outputs.squeeze(0).size()) #torch.Size([4, 400, 640])


        ##PREDICTION
        pred = F.softmax( outputs, dim=1 ) 
        # print(f"pred.size() {pred.size()}") #pred.size() torch.Size([1, 4, 400, 640])

        ## PREDICTION argmax(softmax(x))
        pred_mask = torch.argmax(F.softmax(outputs, dim=1), dim=1) 
        # print(pred_mask.size()) #torch.Size([1, 400, 640])
        # print(pred_mask.squeeze(0).size()) #torch.Size([400, 640])

        ## PRECTION argmax(x)
        outputs_argmax = torch.argmax(outputs[0], dim=1) #print(outputs_argmax.size()) #torch.Size([400, 640])

        ##ONNX model
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image)}
        ort_outs = torch.tensor(  np.asarray(ort_session.run(None, ort_inputs))  )  # print(ort_outs.size()) #torch.Size([1, 1, 4, 400, 640])
        ort_outs = ort_outs.squeeze(0).squeeze(0) # 
        #print(ort_outs.size()) #torch.Size([4, 400, 640])
        ort_outs_argmax = torch.argmax(ort_outs, dim=0) # print(ort_outs_argmax.size())#torch.Size([400, 640])


        #######################################
        ##PLOTTING
        #RAW IMAGE
        ax[0,0].imshow( image.permute(0, 2, 3, 1).squeeze(0).to(torch.long).cpu() )
        ax[0,1].imshow( image.permute(0, 2, 3, 1).squeeze(0)[:,:,0].to(torch.long).cpu() )
        ax[0,2].imshow( image.permute(0, 2, 3, 1).squeeze(0)[:,:,1].to(torch.long).cpu() )
        ax[0,3].imshow( image.permute(0, 2, 3, 1).squeeze(0)[:,:,2].to(torch.long).cpu() )
        # ax[0,1].imshow( image.permute(0, 2, 3, 1).squeeze(0)[:,:,0].to(torch.long).cpu() )
        ax[0,0].set_ylabel('RAW [400,640,3]')
        ax[0,1].set_title('ch0')
        ax[0,2].set_title('ch1')
        ax[0,3].set_title('ch2')

        #MASKS
        # print(label.permute(0, 2, 3, 1).squeeze(0).size()) #torch.Size([400, 640, 4])
        ax[1,0].imshow( label.permute(0, 2, 3, 1).squeeze(0).to(torch.long).cpu()  )
        ax[1,1].imshow( label.permute(0, 2, 3, 1).squeeze(0)[:,:,0].to(torch.long).cpu()  )
        ax[1,2].imshow( label.permute(0, 2, 3, 1).squeeze(0)[:,:,1].to(torch.long).cpu()  )
        ax[1,3].imshow( label.permute(0, 2, 3, 1).squeeze(0)[:,:,2].to(torch.long).cpu()  )
        ax[1,4].imshow( label.permute(0, 2, 3, 1).squeeze(0)[:,:,3].to(torch.long).cpu()  )
        ax[1,0].set_ylabel('labels[400,640,4]')
        ax[1,1].set_title('ch0')
        ax[1,2].set_title('ch1')
        ax[1,3].set_title('ch2')
        ax[1,4].set_title('ch3')

        ##CHANNEL PREDICITONS
        ax[2,0].imshow(pred.permute(0, 2, 3, 1).squeeze(0).detach().cpu())
        ax[2,1].imshow(pred[:,0,:,:].squeeze(0).detach().cpu())
        ax[2,2].imshow(pred[:,1,:,:].squeeze(0).detach().cpu())
        ax[2,3].imshow(pred[:,2,:,:].squeeze(0).detach().cpu())
        ax[2,4].imshow(pred[:,3,:,:].squeeze(0).detach().cpu())
        # print(pred_ch0.size(), type(pred_ch0), pred_ch0.type()) #torch.Size([400, 640]) #<class 'torch.Tensor'> #torch.cuda.FloatTensor
        ax[2,0].set_ylabel('pred [400,640,4]')
        ax[2,1].set_title('ch0')
        ax[2,2].set_title('ch1')
        ax[2,3].set_title('ch2')
        ax[2,4].set_title('ch3')

        ##PREDICTIONS
        # ax[3,0].imshow(pred_mask.squeeze(0).cpu())
        ax[3,5].imshow(pred_mask.squeeze(0).cpu())
        ax[3,5].set_title('argmax(softmax(model(image)))')

        # ##ONNX PREDICTIONS
        #Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-27.269308..12.142393].
        ax[4,0].imshow(ort_outs.permute(1, 2, 0).detach().cpu())
        ax[4,1].imshow(ort_outs.permute(1, 2, 0)[:,:,0].detach().cpu())
        ax[4,2].imshow(ort_outs.permute(1, 2, 0)[:,:,1].detach().cpu())
        ax[4,3].imshow(ort_outs.permute(1, 2, 0)[:,:,2].detach().cpu())
        ax[4,4].imshow(ort_outs.permute(1, 2, 0)[:,:,3].detach().cpu())
        ax[4,5].imshow(ort_outs_argmax.cpu())
        ax[4,0].set_ylabel('onnx [400,640,4]')
        ax[4,1].set_title('ch0')
        ax[4,2].set_title('ch1')
        ax[4,3].set_title('ch2')
        ax[4,4].set_title('ch3')
        ax[4,5].set_title('argmax(onnx)')

        if j == 2:
            break

    plt.show()
