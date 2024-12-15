
import os
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt 
import pandas as pd


if __name__ == "__main__":
    """
    Plot losses from a csv file.
    Arguments:
        -p, --model_path: Set the model path.
        -lf, --loss_file: Set the loss file.
    
    Example:
    python src/ready/apis/plot_losses.py -p <PATH> -lf1 <*.csv> -lf2 <*.csv>
    """

    parser = ArgumentParser(description="Convert models to ONNX and simplify it (sim.onnx)")
    parser.add_argument(
        "-p",
        "--model_path",
        default="none",
        help=("Set the model path"),
    )
    parser.add_argument(
        "-lf1", 
        "--loss_file1", 
        default="none", 
        help=("Set the loss file"))
    parser.add_argument(
        "-lf2", 
        "--loss_file2", 
        default="none", 
        help=("Set the loss file"))

    args = parser.parse_args()
    MODELS_PATH=args.model_path
    loss_file1 = args.loss_file1
    loss_file2 = args.loss_file2

    path_filename1 = os.path.join(MODELS_PATH, loss_file1)
    path_filename2 = os.path.join(MODELS_PATH, loss_file2)
    df1 = pd.read_csv(path_filename1, names=['lf1'], header=None)
    
    df2 = pd.read_csv(path_filename2, names=['lf2'], header=None)
    df1['epochs'] = df1.index
    df1['lf2'] = df2

    plt.plot(df1['epochs'], df1['lf1'],df1['epochs'], df1['lf2'])
    plt.title("Losses for models trianed 100epochs in a100-80gb gpu")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(["Train Loss without agumentation", "Train Loss with agumentation"])
    plt.grid()
    plt.show()