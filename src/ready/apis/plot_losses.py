
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
    
    print(args)
    config_file = args.config_file
    config = OmegaConf.load(config_file)
    MODELS_PATH=os.path.join(Path.home(), config.dataset.models_path)
    TRAINING_LOSS = config.losses.training_loss
    VALIDATION_LOSS = config.losses.validation_loss

    path_training_loss = os.path.join(MODELS_PATH, TRAINING_LOSS)
    path_validation_loss = os.path.join(MODELS_PATH, VALIDATION_LOSS)

    training_loss_df = pd.read_csv(path_training_loss, names=['Training Loss'])
    validation_loss_df = pd.read_csv(path_validation_loss, names=['Validation Loss'])
    
    training_loss_df['epochs'] = training_loss_df.index
    validation_loss_df['epochs'] = validation_loss_df.index

    logger.info(f"\n Training Loss Dataframe: {training_loss_df}")
    logger.info(f"\n Validation Loss Dataframe: {validation_loss_df}")

    # plt.plot(df1['epochs'], df1['lf1'], df1['epochs'], df1['lf2'], df1['epochs'], df1['lf3'], df1['epochs'], df1['lf4'], df1['epochs'], df1['lf5'], linewidth=3)
    plt.plot(training_loss_df['epochs'], training_loss_df['Training Loss'], validation_loss_df['epochs'], validation_loss_df['Validation Loss'], linewidth=3)
    # plt.title("Losses for models trained 100epochs in a100-80gb gpu")

    plt.title("Training and Validation Loss Values")
    plt.xlabel("Epochs", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.tick_params(axis='both', labelsize=13)
    plt.legend(["Training", "Validation"], fontsize=18)
    plt.tight_layout()
    plt.grid()
    plt.show()
