from typing import *
import tensorflow as tf
import keras
import tensorflow_model_optimization as tfmot

def build_cmd(dist_config: dict) -> List[str]:
    cmd = ["python"]
    cmd.append("./worker.py")
    cmd.extend(["--model_type", f"{dist_config['model_type']}"])
    cmd.extend(["--model_idx", f"{dist_config['model_idx']}"])
    cmd.extend(["--input_idx", f"{dist_config['input_idx']}"])
    cmd.extend(["--device_type", f"{dist_config['device_type']}"])
    cmd.extend(["--num_device", f"{dist_config['num_device']}"])
    cmd.extend(["--strategy", f"{dist_config['strategy']}"])
    if 'extra' in dist_config and dist_config["extra"] is not None:
        cmd.extend(["--extra", f"{dist_config['extra']}"])
    return cmd


def generateStrategy(
        num_device: int, strategy: tf.distribute.Strategy, device_type: str = "GPU") -> tf.distribute.Strategy:
    assert (device_type in ["CPU", "GPU"])
    assert (num_device <= len(tf.config.list_logical_devices(device_type)))
    devices = []
    for i in range(num_device):
        devices.append("/{d}:{i}".format(d=device_type, i=i))
    # Create strategy with specific devices
    new_strategy = strategy(
        devices=devices
        # cross_device_ops=cross_device_ops.ReductionToOneDevice()
    )
    return new_strategy

def load_model(model_dir, quantized):
    if quantized:
        with tfmot.quantization.keras.quantize_scope():
            model = keras.models.load_model(f"{model_dir}/model_quantized.h5")
    else:
        model = keras.models.load_model(f"{model_dir}/model.h5")
        
    optimizer = tf.keras.optimizers.SGD(learning_rate=10)
    loss = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.AUTO)
    model.compile(optimizer=optimizer, loss=loss)
    return model
