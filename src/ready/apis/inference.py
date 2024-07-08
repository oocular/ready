"""
Inference
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mimg

from src.ready.utils.datasets import EyeDataset
from src.ready.utils.utils import get_working_directory, set_data_directory
from src.ready.models.unet import UNet

if __name__ == "__main__":
    set_data_directory("datasets/openEDS")
    print(os.getcwd())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainset = EyeDataset("openEDS/openEDS/train/") #train #est #validation
    print("Length of trainset:", len(trainset))

    batch_size_=8 #8 original
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_, shuffle=True, num_workers=4
    )
    print(f"trainloader.batch_size {trainloader.batch_size}")

    checkpoint_path = "weights/trained_models_in_cricket/model-5jul2024.pth"
    model = UNet(nch_in=1, nch_out=4)
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    f, ax = plt.subplots(1, 3)
    cuda_available = torch.cuda.is_available()
    for j, data in enumerate(trainloader, 1):
        print(j)
        images, labels = data
        if cuda_available:
            images = images.cuda()
            labels = labels.cuda()

        #print(images[0].unsqueeze(0).size()) #torch.Size([1, 1, 400, 640])
        #print(labels[0].unsqueeze(0).size()) #torch.Size([1, 400, 640])
        outputs = model(images[0].unsqueeze(0)) #torch.Size([1, 4, 400, 640])
        outputs = torch.argmax(outputs[0], 0) #torch.Size([400, 640])

        #plt.figure()
        ax[0].imshow((images[0] * 255).to(torch.long).squeeze(0).cpu())
        ax[1].imshow(labels[0].squeeze(0).cpu())
        ax[2].imshow(outputs.squeeze(0).cpu())
        #plt.pause(.2) 
        plt.show()

        if j == 5:
             break
 



