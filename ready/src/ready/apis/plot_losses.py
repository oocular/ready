
import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

if __name__ == "__main__":
    """
    Plot losses from a csv file.
    Arguments:
        -c, with model path and loss files

    Example:
    python src/ready/apis/plot_losses.py -c <config.yaml>
    """

    parser = ArgumentParser(description="Plot losses where files are in config file")
    parser.add_argument("-c", "--config_file", help="Config filename with path")
    args = parser.parse_args()
    
    config_file = args.config_file
    config = OmegaConf.load(config_file)
    MODELS_PATH=os.path.join(Path.home(), config.dataset.models_path)
    TRAINING_LOSS_FILES = config.losses.training_loss
    VALIDATION_LOSS_FILES = config.losses.validation_loss 

    for current_training_loss_file in TRAINING_LOSS_FILES:
        
        path_current_training_loss = os.path.join(MODELS_PATH, current_training_loss_file)

        directory, csv_file = os.path.split(current_training_loss_file)
        current_training_loss_df = pd.read_csv(path_current_training_loss, names=[str(current_training_loss_file)])

        current_training_loss_df['epochs'] = current_training_loss_df.index
        
        training_loss_label = "training_loss_" + str(directory) 
        plt.plot(current_training_loss_df['epochs'], current_training_loss_df[str(current_training_loss_file)],label=training_loss_label, linewidth=3) 

    for current_validation_loss_file in VALIDATION_LOSS_FILES:

        path_current_validation_loss = os.path.join(MODELS_PATH, current_validation_loss_file)

        directory, csv_file = os.path.split(current_validation_loss_file)
        current_validation_loss_df = pd.read_csv(path_current_validation_loss, names=[str(current_validation_loss_file)])

        current_validation_loss_df['epochs'] = current_validation_loss_df.index

        validation_loss_label = "validation_loss_" + str(directory)
        plt.plot(current_validation_loss_df['epochs'], current_validation_loss_df[str(current_validation_loss_file)], label=validation_loss_label, linewidth=3, linestyle='dashed')
    
    plt.title("Training and Validation Loss for each Epoch", fontsize=18)
    plt.xlabel("Epochs", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.tick_params(axis='both', labelsize=13) 
    plt.legend(framealpha=0.5)
    plt.grid()
    plt.show()
