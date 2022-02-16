import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='3' # disable tensorflow logs
warnings.filterwarnings('ignore')

import argparse
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10

from data import augmentation as F
from data.data_loader import DataLoader, DataLoaderTransform
from models.model import build_model
from utils.helper import load_config, plot_history_from_df, set_seed


mpl.rcParams['figure.figsize'] = (10, 8)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
start_time = time.time()


class Trainer(object):

    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loss_monitor = tf.keras.metrics.Mean(name="loss") # batch内の要素で／trackしたものを 平均するのでMean
        self.val_loss_monitor = tf.keras.metrics.Mean(name="val_loss") # batch内の要素で／trackしたものを 平均するのでMean
        self.train_acc_monitor = tf.keras.metrics.CategoricalAccuracy(name='acc') # takes one-hot labels and prediction
        self.val_acc_monitor = tf.keras.metrics.CategoricalAccuracy(name='val_acc')

    @tf.function
    def train_step(self, x: np.ndarray, y: np.ndarray) -> dict:
        """
        Return the average of loss and acc from the start of epoch to that step.
        """
        with tf.GradientTape() as tape:
            softmax_score_pred = self.model(x, training=True)
            loss = self.loss_fn(y, softmax_score_pred)  # y = not one-hot encoded label, but class num
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.train_loss_monitor.update_state(loss)
        self.train_acc_monitor.update_state(y, softmax_score_pred)
        return {"loss": self.train_loss_monitor.result(), "acc": self.train_acc_monitor.result()}

    @tf.function
    def test_step(self, x: np.ndarray, y: np.ndarray) -> dict:
        softmax_score_pred = self.model(x, training=False)
        val_loss = self.loss_fn(y, softmax_score_pred)  # y = not one-hot encoded label, but class num
        self.val_loss_monitor.update_state(val_loss)
        self.val_acc_monitor.update_state(y, softmax_score_pred)
        return {"val_loss": self.val_loss_monitor.result(), "val_acc": self.val_acc_monitor.result()}

    @tf.function
    def test_step_with_tta_loop(self, x: np.ndarray, y: np.ndarray, augmenter_list: list) -> dict:
        """
        Merge method: Argmax of average of softmax probabilities
        """
        # Compute validation loss fist
        softmax_score_pred = self.model(x, training=False)
        val_loss = self.loss_fn(y, softmax_score_pred)  # y = not one-hot encoded label, but class num
        self.val_loss_monitor.update_state(val_loss)
        # TTA loop to get an accuracy
        val_softmax_list = [softmax_score_pred]
        for augmenter in augmenter_list: # perform predictions for the number of augmentations
            softmax_score_pred = self.model(augmenter(x), training=False)
            val_softmax_list.append(softmax_score_pred)
        val_softmax_array = tf.reduce_mean(val_softmax_list, axis=0) # (batch_size, 10)
        self.val_acc_monitor.update_state(y, val_softmax_array)
        return {"val_loss": self.val_loss_monitor.result(), "val_acc": self.val_acc_monitor.result()}


def evaluate(model, test_ds, metrics, tta, augmenter_list):
    for x, y in test_ds:
        if tta:
            softmax_score = model(x, training=False)
            softmax_score_list = [softmax_score]
            for augmenter in augmenter_list:
                softmax_score_list.append(model(augmenter(x), training=False))
            softmax_score = np.mean(softmax_score_list, axis=0)
        else:
            softmax_score = model(x, training=False)

        for metric in metrics:
            metric.update_state(y, softmax_score)


def main(args: argparse.ArgumentParser):
    # Setup
    # GPU configuration
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices):
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print('memory growth:', tf.config.experimental.get_memory_growth(device))
    else:
        print('Not enough GPU hardware devices available')
    # General configration
    config = load_config(f'config/{args.config}', args)
    if os.getenv('RUN_ID') is not None:
        config.run_id = f'{args.config.replace(".yaml", "")}_{os.getenv("RUN_ID")}'
    else:
        config.run_id = f'{args.config.replace(".yaml", "")}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    if config.debug:
        config.run_id = 'debug_' + config.run_id
    if config.seed is not None:
        set_seed(int(config.seed))
    else:
        set_seed(config.random_seed)

    print(f'Eager execution: {tf.executing_eagerly()}')
    print(f'TensorFlow version: {tf.__version__}')
    print(f'Eager execution: {tf.executing_eagerly()}')
    print(f'Debugging mode: {config.debug}')
    print(f'AUG: {config.aug}')
    print(f'TTA: {config.tta}')
    augmenter_list = [F.horizontal_flip_np]
    if config.debug:
        config.epochs = 5
    if not args.not_record:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base_dir, config.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        ckpt_dir = os.path.join(base_dir, config.ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        report_dir = os.path.join(base_dir, config.report_dir)
        os.makedirs(report_dir, exist_ok=True)
        history_dir = os.path.join(base_dir, config.report_dir, 'history')
        os.makedirs(history_dir, exist_ok=True)
        meta_dir = os.path.join(base_dir, config.report_dir, 'meta')
        os.makedirs(meta_dir, exist_ok=True)


    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=config.test_size, shuffle=True, random_state=config.random_seed)
    x_train = np.float32(x_train) / 255.
    x_valid = np.float32(x_valid) / 255.
    x_test = np.float32(x_test) / 255.
    y_train = tf.keras.utils.to_categorical(y_train)
    y_valid = tf.keras.utils.to_categorical(y_valid)
    y_test = tf.keras.utils.to_categorical(y_test)
    print(y_train.shape)
    print(y_valid.shape)
    print(f'Inputs : train: {x_train.shape}, valid: {x_valid.shape}, test: {x_test.shape}')
    print(f'Targets: train: {y_train.shape}, valid: {y_valid.shape}, test: {y_test.shape}')
    if config.debug:
        x_train, y_train = x_train[:len(x_train) // 10], y_train[:len(x_train) // 10]
        x_valid, y_valid = x_valid[:len(x_valid) // 10], y_valid[:len(x_valid) // 10]
        x_test, y_test = x_test[:len(x_test) // 10], y_test[:len(x_test) // 10]


    # Construct Dataset / Data Loader
    if config.aug:
        train_data_gen = DataLoaderTransform(augmenter_list, 0.5, x_train, y_train, config.batch_size, shuffle=True)
    else:
        train_data_gen = DataLoader(x_train, y_train, config.batch_size, shuffle=True)
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(config.batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(config.batch_size)
    print('DataLoader for train:', train_data_gen)


    # Build model and trainer
    model = build_model()
    model.summary()
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False) # logits = Not softmax applied prediction
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False) # logits = not softmax applied prediction
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.initial_lr)
    trainer = Trainer(model, loss_fn, optimizer)


    loss_list, acc_list = [], []
    val_loss_list, val_acc_list = [], []
    # Training loop
    for epoch in range(config.epochs):
        epoch_enter = time.time()
        print('-----------')

        # learning rate decay at 80 epoch
        if epoch == 69:
            optimizer.lr.assign(config.initial_lr / 5)

        # Training loop
        for step in range(len(train_data_gen)):
            x_batch_train, y_batch_train = train_data_gen[step]
            metrics = trainer.train_step(x_batch_train, y_batch_train)
            print(f'\rEpoch {epoch+1}/{config.epochs}: ({step+1}/{len(train_data_gen)}) -- loss: {float(metrics["loss"]):.4f} - acc: {float(metrics["acc"]):.4f}', end='')

        # Append and reset training metrics at the end of each epoch
        loss_list.append(float(trainer.train_loss_monitor.result()))
        acc_list.append(float(trainer.train_acc_monitor.result()))
        trainer.train_loss_monitor.reset_states()
        trainer.train_acc_monitor.reset_states()

        # Validation loop at the end of each epoch
        for x_batch_val, y_batch_val in valid_dataset:
            if config.tta:
                _ = trainer.test_step_with_tta_loop(x_batch_val, y_batch_val, augmenter_list)
            else:
                _ = trainer.test_step(x_batch_val, y_batch_val)
        val_loss = trainer.val_loss_monitor.result()
        val_acc = trainer.val_acc_monitor.result()
        val_loss_list.append(float(val_loss))
        val_acc_list.append(float(val_acc))
        print(f'\rEpoch {epoch+1}/{config.epochs}: ({step+1}/{len(train_data_gen)}) -- loss: {float(metrics["loss"]):.4f} - acc: {float(metrics["acc"]):.4f}'
                + f' - val_loss: {float(val_loss):.4f} - val_acc: {float(val_acc):.4f} ' , end='')
        trainer.val_loss_monitor.reset_states()
        trainer.val_acc_monitor.reset_states()
        print('\nTime taken: {:.2f}s'.format(time.time() - epoch_enter))


    # Summarize
    history_df = pd.DataFrame({'loss': loss_list, 'val_loss': val_loss_list, 'acc': acc_list, 'val_acc': val_acc_list})
    history_df['epoch'] = list(range(1, config.epochs + 1))
    history_df.set_index('epoch', inplace=True)
    # Save outputs
    if not args.not_record:
        model.save(os.path.join(ckpt_dir, f'ckpt_{config.run_id}.h5')) # ckpt
        history_df.to_csv(os.path.join(history_dir, f'history_{config.run_id}.csv')) # learning curve
        plot_history_from_df(history_df, filename=f'{log_dir}/log_{config.run_id}.png', suptitle=config.config.replace('.yaml', ''), show=False)


    # Evaluate model
    test_metrics = [
        tf.keras.metrics.CategoricalCrossentropy(name='test_loss'),
        tf.keras.metrics.CategoricalAccuracy(name='test_acc')
    ]
    evaluate(model, test_dataset, test_metrics, config.tta, augmenter_list)
    print(f'Test loss: {test_metrics[0].result()}')
    print(f'Test accuracy: {test_metrics[1].result()}')


    report = {
        'test_loss_score': float(test_metrics[0].result()),
        'test_acc_score': float(test_metrics[1].result()),
    }
    if not args.not_record:
        # Dump config and test score
        with open(os.path.join(meta_dir, f'meta_{config.run_id}.json'), 'w') as f:
            meta = dict([(k, v) for k, v in config.items() if not isinstance(v, np.ndarray)])
            meta = {**meta, **report}
            augmenter_names = [a.__name__ for a in augmenter_list]
            meta.update(augmentations_used=augmenter_names)
            json.dump(meta, f, indent=4)
    print(f'Time taken for all: {time.time() - start_time:.2f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', type=str, default='01_no_aug_no_tta.yaml', help='Config filename')
    parser.add_argument('-d', '--debug', action='store_true', help='Debugging mode')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU id:=0')
    parser.add_argument('-nr', '--not-record', action='store_true', help='Not record logs')
    parser.add_argument('-s', '--seed', type=str, default=None, help='Enable online logging')
    parser.add_argument('-v', '--verbose', type=int, default=1, choices=[0, 1, 2], help='Verbosity level')
    parser.add_argument('-w', '--wandb', action='store_true', help='Enable online logging')
    args = parser.parse_args()

    main(args)
