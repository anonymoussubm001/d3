import os
import keras
import logging
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from keras import Model, layers
from constants import *
from convert_to_sync_batchnorm import get_models_with_batchnorm

tf.get_logger().setLevel(logging.ERROR)

# List of layers that can be quantized
quantizable_layers = [
    layers.Conv1D,
    layers.Conv2D,
    layers.Conv3D,
    layers.DepthwiseConv1D,
    layers.DepthwiseConv2D,
    layers.Dense,
    layers.ReLU,
    layers.Concatenate,
    layers.Add,
]


def apply_quantization(layer):
  if type(layer) in quantizable_layers:
    return tfmot.quantization.keras.quantize_annotate_layer(layer)
  return layer


if __name__ == "__main__":
    print("Generating quantized models")
    cur_dir = os.getcwd()
    model_dir = f"{cur_dir}/../data/models"
    models_with_batchnorm = get_models_with_batchnorm(model_dir)
    for m in range(NUM_MODELS//2, NUM_MODELS):
        try:
            if m in models_with_batchnorm:
                original_model: Model = keras.models.load_model(f"{model_dir}/model_{m}/model_syncbatch.h5")
            else:
                original_model: Model = keras.models.load_model(f"{model_dir}/model_{m}/model.h5")
            annotated_model = tf.keras.models.clone_model(original_model, clone_function=apply_quantization)

            converted_model = tfmot.quantization.keras.quantize_apply(annotated_model)
            keras.models.save_model(converted_model, f"{model_dir}/model_{m}/model_quantized.h5")
            print(f"Model {m} converted")
        except Exception as e:
            print(str(e))
            print(f"Model {m} conversion failed")
