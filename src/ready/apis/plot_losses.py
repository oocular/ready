
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
    parser.add_argument("-c", "--config_file", help="Config filename with path", type=str)
    args = parser.parse_args()

    config_file = args.config_file
    config = OmegaConf.load(config_file)
    MODELS_PATH=os.path.join(Path.home(), config.dataset.models_path)

    path_filename1 = os.path.join(MODELS_PATH, config.losses.loss_file1)
    path_filename2 = os.path.join(MODELS_PATH, config.losses.loss_file2)
    path_filename3 = os.path.join(MODELS_PATH, config.losses.loss_file3)
    path_filename4 = os.path.join(MODELS_PATH, config.losses.loss_file4)
    path_filename5 = os.path.join(MODELS_PATH, config.losses.loss_file5)

    df1 = pd.read_csv(path_filename1, names=['lf1'], header=None)
    df2 = pd.read_csv(path_filename2, names=['lf2'], header=None)
    df3 = pd.read_csv(path_filename3, names=['lf3'], header=None)
    df4 = pd.read_csv(path_filename4, names=['lf4'], header=None)
    df5 = pd.read_csv(path_filename5, names=['lf5'], header=None)

    df1['epochs'] = df1.index
    df1['lf2'] = df2
    df1['lf3'] = df3
    df1['lf4'] = df4
    df1['lf5'] = df5

    logger.info(f"\n Dataframe: {df1}")

    # plt.plot(df1['epochs'], df1['lf1'], df1['epochs'], df1['lf2'], df1['epochs'], df1['lf3'], df1['epochs'], df1['lf4'], df1['epochs'], df1['lf5'], linewidth=3)
    plt.plot(df1['epochs'], df1['lf1'], df1['epochs'], df1['lf2'], df1['epochs'], df1['lf3'], linewidth=3)
    # plt.title("Losses for models trained 100epochs in a100-80gb gpu")

    plt.xlabel("Epochs", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.tick_params(axis='both', labelsize=13)
    # plt.legend(["naug_d1144", "waug_d1144", "waug_d0572", "waug_d0286", "waug_d0145"])
    plt.legend(["naug_d1144", "waug_d1144", "waug_d0572"], fontsize=18)
    plt.tight_layout()
    plt.grid()
    plt.show()
