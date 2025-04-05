
import os
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np

import matplotlib.pyplot as plt
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

    file_numbers = range(1, 6)
    path_filenames = [
        os.path.join(MODELS_PATH, getattr(config.json_files, f"file{i}"))
        for i in file_numbers
    ]
    print(path_filenames[0])

    file_paths = [path_filenames[0], path_filenames[1], path_filenames[2]]
    # file_paths = [path_filenames[0], path_filenames[1], path_filenames[2], path_filenames[3], path_filenames[4]]
    logger.info(f"\n Files: {file_paths}")

    data_list = []

    for file_path in file_paths:
        with open(file_path) as f:
            data_list.append(json.load(f))
    labels = list(data_list[0].keys())
    # labels = labels[:-1] #remove dice TODO > #https://github.com/oocular/ready/issues/77

    model_names = ["naug_d1144", "waug_d1144", "waug_d0572"] #waug_d0286 #waug_d0145
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(labels))
    bar_width = 0.15

    for i, data in enumerate(data_list):
        values = [data[label] for label in labels]
        offset = bar_width * i
        rects = ax.bar(x + offset, values, bar_width, label=model_names[i] )

    # ax.set_title('Comparison of Model Metrics', fontsize=14)
    ax.set_xlabel('Metrics', fontsize=18)
    ax.set_ylabel('Scores', fontsize=18)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_xticks(x + bar_width * (len(data_list)-1)/2)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.5)
    ax.set_ylim(0.99, 1.0)

    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.show()
