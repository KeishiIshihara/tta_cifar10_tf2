import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import yaml
import seaborn as sns
from argparse import ArgumentParser
from dotmap import DotMap
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


def load_config(config_path: str, args: ArgumentParser=None):
    with open(config_path, 'r+') as yaml_file:
        config = yaml.load(yaml_file, Loader)
    if args is not None:
        config = {**config, **vars(args)}
    return DotMap(config)


def plot_history(hist, filename=None):
    sns.set()
    sns.set_style('whitegrid')
    # Losses and metrics
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    # Epochs to plot along x axis
    x_axis = range(1, len(loss) + 1)
    # figure
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
    # Axis 1
    ax1.plot(x_axis, loss, label='Training')
    ax1.plot(x_axis, val_loss, label='Validation')
    ax1.set_title('MSE Loss')
    ax1.legend()
    # Axis 2
    ax2.plot(x_axis, acc, label='Training')
    ax2.plot(x_axis, val_acc, label='Validation')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.legend()
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)


def plot_history_from_df(history_df: pd.DataFrame, filename: str=None, suptitle: str=None, show: bool=False):
    sns.set()
    sns.set_style('whitegrid')
    cargs = {'kind':'line', 'use_index':True, 'grid': True, 'legend': True}
    args_list = [
        {'title': 'Loss', 'column': ['loss', 'val_loss']},
        {'title': 'Accuracy', 'column': ['acc', 'val_acc'], 'ylim': [0.0, 1.0]},
    ]
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(10, 5), sharex=True)
    for ax, args in zip(axes.ravel(), args_list):
        history_df[args['column']].plot(ax=ax,  **cargs, title=args['title'])
        ax.set_ylabel(args['title'])
    if suptitle is not None:
        fig.suptitle(suptitle)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()
