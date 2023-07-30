from dlrm.dlrm import DLRM
from test_utils import *
from constants import *
from keras import optimizers
from tensorflow import Tensor
from typing import *
import tensorflow as tf
import pickle
import json
import os
import tensorflow_model_optimization as tfmot


class DLRMModelTest():
    """
    Class for testing DLRM models
    """

    def __init__(self, model_idx: int, input_idx: int, strategy: tf.distribute.Strategy, dist_setting_str: str,
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
        with open(f"{self.model_dir}/model_spec.json", 'r') as f:
            self.model_specs = json.load(f)

    def DLRM_train_nondist(self,
                           train_dataset: tf.data.Dataset,
                           test_dataset: tf.data.Dataset) -> Tensor:

        # Load model
        model: DLRM = DLRM(
            num_embed=self.model_specs["num_embed"],
            embed_dim=self.model_specs["embed_dim"],
            embed_vocab_size=self.model_specs["embed_vocab_size"],
            ln_bot=self.model_specs["ln_bot"],
            ln_top=self.model_specs["ln_top"])
        model.load_weights(f"{self.model_dir}/weights/")
        optimizer = optimizers.Adam()

        # Make one inference to build the model (making quantization possible)
        for data in test_dataset:
            _ = model.inference(data["dense_features"], data["sparse_features"])
            break

        if self.model_extra == "quantize":
            model._mlp_bot = tfmot.quantization.keras.quantize_model(model._mlp_bot)
            model._mlp_top = tfmot.quantization.keras.quantize_model(model._mlp_top)

        def train_step(dense_features: Tensor, sparse_features: Tensor, label: Tensor):
            with tf.GradientTape() as tape:
                loss_value = model.get_myloss(
                    dense_features, sparse_features, label)
            gradients = tape.gradient(
                loss_value, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))
            return loss_value

        # Train for 1 step
        for data in train_dataset:
            loss: Tensor = train_step(
                data["dense_features"], data["sparse_features"], data["label"])

        # Test
        for data in test_dataset:
            pred: Tensor = model.inference(
                data["dense_features"], data["sparse_features"])

        return pred

    def DLRM_train_dist(self,
                        strategy: tf.distribute.Strategy,
                        train_dataset: tf.data.Dataset,
                        test_dataset: tf.data.Dataset) -> Tensor:

        train_dataset_dist = strategy.experimental_distribute_dataset(
            train_dataset)
        test_dataset_dist = strategy.experimental_distribute_dataset(
            test_dataset)

        # Test
        @tf.function
        def distributed_inference(dist_inputs: tf.data.Dataset):
            per_replica_prediction = strategy.run(model_dist.inference, args=(
                dist_inputs["dense_features"], dist_inputs["sparse_features"]))
            return per_replica_prediction

        # Load model
        with strategy.scope():
            model_dist: DLRM = DLRM(
                num_embed=self.model_specs["num_embed"],
                embed_dim=self.model_specs["embed_dim"],
                embed_vocab_size=self.model_specs["embed_vocab_size"],
                ln_bot=self.model_specs["ln_bot"],
                ln_top=self.model_specs["ln_top"])

            model_dist.load_weights(f"{self.model_dir}/weights/")
            optimizer = optimizers.SGD(learning_rate=10)

            for dist_inputs in test_dataset_dist:
                _ = distributed_inference(dist_inputs)
                break

            if self.model_extra == "quantize":
                model_dist._mlp_bot = tfmot.quantization.keras.quantize_model(model_dist._mlp_bot)
                model_dist._mlp_top = tfmot.quantization.keras.quantize_model(model_dist._mlp_top)

        def train_step(dense_features: Tensor, sparse_features: Tensor, label: Tensor):
            with tf.GradientTape() as tape:
                loss_value = model_dist.get_myloss_dist(
                    dense_features, sparse_features, label, BATCH_SIZE)
            gradients = tape.gradient(
                loss_value, model_dist.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model_dist.trainable_variables))
            return loss_value

        @tf.function
        def distributed_train_step(dist_inputs):
            per_replica_losses = strategy.run(train_step, args=(
                dist_inputs["dense_features"], dist_inputs["sparse_features"], dist_inputs["label"]))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

        # Train for 1 step
        for dist_inputs in train_dataset_dist:
            loss: Tensor = distributed_train_step(dist_inputs)

        for dist_inputs in test_dataset_dist:
            pred = distributed_inference(dist_inputs)

        if strategy.num_replicas_in_sync > 1:
            pred = tf.concat(pred.values, axis=0)

        return pred

    def test_DLRM_train(self):
        # Sharding policy options for datasets
        options = tf.data.Options()
        options.deterministic = True
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        # Load test data
        with open(f"{self.input_dir}/test_{self.input_idx}.pk", 'rb') as f:
            (test_dense, test_sparse, test_label) = pickle.load(f)
        test_dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices({
            'dense_features': test_dense,
            'sparse_features': test_sparse,
            'label': test_label
        }).batch(BATCH_SIZE).with_options(options)

        # Load training data
        with open(f"{self.input_dir}/train_{self.input_idx}.pk", 'rb') as f:
            (train_dense, train_sparse, train_label) = pickle.load(f)
        train_dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices({
            'dense_features': train_dense,
            'sparse_features': train_sparse,
            'label': train_label
        }).batch(BATCH_SIZE).with_options(options)

        num_devices = int(self.dist_setting_str[0])

        # One step train and test
        if num_devices == 0:
            pred = self.DLRM_train_nondist(train_dataset, test_dataset)
        else:
            pred = self.DLRM_train_dist(self.strategy, train_dataset, test_dataset)

        if self.model_extra == "quantize":
            suffix = "_q"
        else:
            suffix = ""

        # Save output
        with open(f"{self.result_dir}/output_{self.input_idx}_{self.dist_setting_str}{suffix}.pk", 'wb') as f:
            pickle.dump(pred, f)
