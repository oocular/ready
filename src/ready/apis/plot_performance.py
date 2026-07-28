
import json
import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import OmegaConf

if __name__ == "__main__":
    """
    Plot losses from a csv file.
    Arguments:
        -c, with model path and loss files

    Example:
    python src/ready/apis/plot_performance.py -c <config.yaml>
    """

    parser = ArgumentParser(description="Plot losses where files are in config file")
    parser.add_argument("-c", "--config_file", help="Config filename with path", type=str)
    args = parser.parse_args()

    config_file = args.config_file
    config = OmegaConf.load(config_file)
    MODELS_PATH=os.path.join(Path.home(), config.dataset.models_path)
    TRAINING_PERFORMANCE_FILES = config.performance.training_performance
    VALIDATION_PERFORMANCE_FILES = config.performance.validation_performance
    
    performance_metrics = [
        "accuracy",
        "f1",
        "recall",
        "precision",
        "fbeta",
        "miou",
        "dice"
    ]

    x_axis = np.arange(len(performance_metrics))
    n_bars = len(TRAINING_PERFORMANCE_FILES)
    width = 0.3/n_bars
    for i, current_training_performance_file in enumerate(TRAINING_PERFORMANCE_FILES, start=1):

        path_current_training_performance = os.path.join(MODELS_PATH, current_training_performance_file)

        directory, _ = os.path.split(current_training_performance_file)
        
        current_training_performance = pd.read_json(path_current_training_performance, typ='series')
        current_training_performance_values = current_training_performance.array
        
        current_training_performance_label = "training_performance_" + str(directory)
        plt.bar(x_axis - (i - (n_bars - 1) / 2) * width, current_training_performance_values, width=width, label=current_training_performance_label, alpha=0.5)

    
    for i, current_validation_performance_file in enumerate(VALIDATION_PERFORMANCE_FILES, start=1):
        
        path_current_validation_performance = os.path.join(MODELS_PATH, current_validation_performance_file)

        directory, _ = os.path.split(current_validation_performance_file)

        current_validation_performance = pd.read_json(path_current_validation_performance, typ='series')
        current_validation_performance_values = current_validation_performance.array

        current_validation_performance_label = "validation_performance_" + str(directory)
        plt.bar(x_axis + (i - (n_bars -1 )/2) * width, current_validation_performance_values, width=width, label=current_validation_performance_label, alpha=0.5) 
     
    plt.xticks(x_axis, performance_metrics)
    plt.ylabel('Values', fontsize=18)
    plt.xlabel('Performance Metrics', fontsize=18)
    plt.title('Training and Validation Performance Metrics', fontsize=18)
    plt.legend(fontsize=15, loc='center right', framealpha=0.5)
    plt.tick_params(axis='both', labelsize=17)
    plt.grid(visible=True)
    plt.show()
