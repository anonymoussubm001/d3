from keras import Model
from test_utils import *
from constants import *
from tensorflow import Tensor
from constants import *
import tensorflow as tf
import pickle
import os


class SequentialModelTest():
    def __init__(
            self, model_idx: int, input_idx: int, strategy: tf.distribute.Strategy, dist_setting_str: str,
            model_extra: str):
        self.model_idx = model_idx
        self.input_idx = input_idx
        self.strategy = strategy
        self.dist_setting_str = dist_setting_str
        self.model_extra = model_extra
        cur_dir = os.getcwd()
        self.model_dir = f"{cur_dir}/../data/models/model_{model_idx}"
        self.input_dir = f"{cur_dir}/../data/inputs/input_{model_idx}"
        self.result_dir = f"{cur_dir}/../results/outputs/output_{model_idx}"

    def Sequential_train_nondist(self,
                                 train_dataset: tf.data.Dataset,
                                 test_dataset: tf.data.Dataset) -> Tensor:
        # Load model
        quantized = (self.model_extra == "quantized")
        model: Model = load_model(self.model_dir, quantized)

        # Train for 1 step
        loss: Tensor = model.fit(
            train_dataset[0], train_dataset[1], verbose=0, shuffle=False, batch_size=BATCH_SIZE)

        pred: Tensor = model(test_dataset[0])

        return pred

    def Sequential_train_dist(self,
                              strategy: tf.distribute.Strategy,
                              train_dataset: tf.data.Dataset,
                              test_dataset: tf.data.Dataset) -> Tensor:
        # Load model
        with strategy.scope():
            # Load model
            quantized = (self.model_extra == "quantized")
            model: Model = load_model(self.model_dir, quantized)

        # Train for 1 step
        loss: Tensor = model.fit(
            train_dataset[0], train_dataset[1], verbose=0, shuffle=False, batch_size=BATCH_SIZE)

        pred: Tensor = model(test_dataset[0])

        return pred

    def test_Sequential_train(self):
        # Load test data
        with open(f"{self.input_dir}/test_{self.input_idx}.pk", 'rb') as f:
            (test_input, test_label) = pickle.load(f)

        # Load training data
        with open(f"{self.input_dir}/train_{self.input_idx}.pk", 'rb') as f:
            (train_input, train_label) = pickle.load(f)

        num_devices = int(self.dist_setting_str[0])

        # One step train and test
        if num_devices == 0:
            pred = self.Sequential_train_nondist(
                train_dataset=(train_input, train_label),
                test_dataset=(test_input, test_label))
        else:
            pred = self.Sequential_train_dist(
                strategy=self.strategy,
                train_dataset=(train_input, train_label),
                test_dataset=(test_input, test_label))

        if self.model_extra == "quantize":
            suffix = "_q"
        else:
            suffix = ""

        # Save output
        with open(f"{self.result_dir}/output_{self.input_idx}_{self.dist_setting_str}{suffix}.pk", 'wb') as f:
            pickle.dump(pred, f)
