
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
    TRAINING_PERFORMANCE = config.performance.training_performance
    VALIDATION_PERFORMANCE = config.performance.validation_performance
    
    path_training_performance = os.path.join(MODELS_PATH, TRAINING_PERFORMANCE) 
    path_validation_performance = os.path.join(MODELS_PATH, VALIDATION_PERFORMANCE)
    
    training_performance = pd.read_json(path_training_performance, typ='series')
    validation_performance = pd.read_json(path_validation_performance, typ='series')

    logger.info(f"Training performance metrics: {training_performance}")
    logger.info(f"Validation performance metrics: {validation_performance}")

    performance_metrics = [
        "accuracy",
        "f1",
        "recall",
        "precision",
        "fbeta",
        "miou",
        "dice"
    ]
    
    training_performance_values, validation_performance_values = training_performance.array, validation_performance.array
    width, x_axis = 0.3, np.arange(len(performance_metrics))
    
    plt.bar(x_axis - width/2, training_performance_values, width, label='Training')
    plt.bar(x_axis + width/2, validation_performance_values, width, label='Validation')

    plt.xticks(x_axis, performance_metrics)
    plt.ylabel('Values', fontsize=18)
    plt.xlabel('Performance Metrics', fontsize=18)
    plt.title('Training and Validation Performance Metrics: 27-Jul-2025_03-44-52_NVIDIA_A100_80GB_PCI', fontsize=18)
    plt.legend(fontsize=18, loc='center right', framealpha=0.5)
    plt.tick_params(axis='both', labelsize=17)
    plt.grid(visible=True)
    plt.tight_layout()
    plt.show()
