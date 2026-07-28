#from flwr.server.strategy import FedAvg
from ready.models.unet import UNet
import pathlib
import torch
import torch.nn as nn
from ready.apis.train_federated_mobious import main as train_federated_mobious
import os
import subprocess
#from ready.apis.train_federated_rti_eyes import main as train_federated_rti_eyes
#from loguru import logger
from argparse import Namespace
from loguru import logger

#import numpy
#import torchvision

#model = UNet(nch_in=3, nch_out=4, nch_ker=64)



def fedAvg(dataset_weights, dataset_sizes):
    #dataset_sizes = []
    #dataset_weights = []

    total_sizes = sum(dataset_sizes)

    average_weights = {}

    for key in dataset_weights[0].keys():
        # Weighted sum across all clients
        average_weights[key] = sum(dataset_weights[i][key] * dataset_sizes[i]
            for i in range(len(dataset_weights))) / total_sizes
    
    return average_weights

def federated(Num_of_rounds, weights):
    
    model = UNet(nch_in=3, nch_out=4, nch_ker=64)


    for round in range(Num_of_rounds):

        #args_rti = Namespace(config_file="configs/federated/config_federated_rti_eyes.yaml")
        #train_federated_rti_eyes(args_rti)

        #os.system("bash scrips/federated/train_federated_openEDS.bash")
        subprocess.run(["bash", "scripts/federated/train_federated_openEDS.bash"])

        args_mobious = Namespace(config_file="configs/federated/config_federated_mobious.yaml")
        train_federated_mobious(args_mobious)

        mobious_weights = torch.load(weights /"mobious_weights.pth")
        openEDS_weights = torch.load(weights /"openEDS_weights.pth")
        #rti_eyes_weights = torch.load(weights /"rti_eyes_weights.pth")

        mobious_size = 3559
        openEDS_size = 27431
        #rti_eyes_size = 8000

        dataset_weights = [mobious_weights, openEDS_weights] #rti_eyes_weights]
        #dataset_weights = [rti_eyes_weights]
        dataset_sizes = [mobious_size, openEDS_size] #rti_eyes_size]
        #dataset_sizes = [rti_eyes_size]


        new_global_model = fedAvg(dataset_weights, dataset_sizes)
        mobious_weights_path = os.path.join(pathlib.Path.home(), "Scratch/scratch/ccaekqu/datasets/ready/ready/mobious/models/15-Jul-2026_12-01-38_cpu/mobious_weights.pth")
        #mobious_weights_path = os.path.join(pathlib.Path.home(), "downloads/ready/datasets/ready/mobious/models/15-Jul-2026_12-01-38_cpu/mobious_weights.pth")
        os.makedirs(os.path.dirname(mobious_weights_path), exist_ok=True)
        torch.save(new_global_model, mobious_weights_path)


        print(f"Federated learning complete — {Num_of_rounds} rounds finished")
        print(f"Global model saved to: {mobious_weights_path}")

if __name__ == "__main__":
    #torch.set_num_threads(1)    #reduce number of processing threads to avoid deadlocks when using DataLoader with num_workers > 0
    #torch.set_num_interop_threads(1)
    #torch.backends.mkldnn.enabled = False    

    weights_path = os.path.join(pathlib.Path.home(), "Scratch/scratch/ccaekqu/datasets/ready/ready/federated")
    #weights_path = "downloads/ready/datasets/ready/federated"
    weights = pathlib.Path(weights_path)
    
    federated(10, weights)
    logger.info(f"##################   DONE #############")
