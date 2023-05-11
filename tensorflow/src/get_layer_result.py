import tensorflow as tf
import argparse
import pickle
import random
import os
import sys
import numpy as np
from constants import *
from keras import backend as K
from test_utils import generateStrategy, load_model


def layer_output(model_id: int, input_id: int, device: str):
    # Get all layer's intermediate output from a (model, input, setting) pair
    # NOTE: the output may occupy large stroage space (~200GB). 
    # Consider reducing BATCH_SIZE in constants.py if limited disk space is available
    cur_dir = os.getcwd()
    model_dir = f"{cur_dir}/../data/models/model_{model_id}/"
    input_dir = f"{cur_dir}/../data/inputs/input_{model_id}/"
    out_dir = f"{cur_dir}/../results/analysis/output_{model_id}_{input_id}/"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_file = out_dir + f"{device}.txt"

    num_device = int(device[0])
    quantized = device.endswith("_q")
    type_device = device[1:].replace("_q", "")

    if num_device == 0:
        model = load_model(model_dir, quantized=quantized)
    else:
        strategy = generateStrategy(
            num_device=num_device, strategy=tf.distribute.MirroredStrategy, device_type=type_device)

        with strategy.scope():
            model = load_model(model_dir, quantized=quantized)

    # Create a function for each layer output
    inp = model.input
    outputs = [layer.output for layer in model.layers]
    functors = [K.function([inp], [out]) for out in outputs]

    with open(input_dir + f"train_{input_id}.pk", 'rb') as f:
        (train_input, train_label) = pickle.load(f)
    model.fit(train_input, train_label, verbose=0, shuffle=False, batch_size=BATCH_SIZE)

    with open(input_dir + f"test_{input_id}.pk", 'rb') as f:
        (test_input, _) = pickle.load(f)

    data_out = []

    with open(out_file, 'w') as f:
        test = test_input
        layer = 0
        for func in functors:
            f.write(f"Layer {layer} Name: {model.layers[layer]}")
            f.write('\n')
            res = func([test])
            f.write(f"{res}")
            f.write('\n')

            # Save output
            data_out.append(np.array(res).squeeze())
            layer += 1

        after_normalize = tf.nn.softmax(data_out[-1])
        f.write("After normalization:\n")
        f.write(f"{after_normalize}")

    with open(f"{out_dir}/output_{device}.pk", 'wb') as f:
        pickle.dump(data_out, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, choices=["CPU", "GPU"],default="CPU")
    parser.add_argument("--infile", type=str, default="../results/csv/inconsistency.csv")

    args = parser.parse_args()
    config = vars(args)

    device = config["device"].upper()
    incons_csv = config["infile"]

    # Set random seeds
    seed = 0
    random.seed(seed)
    tf.random.set_seed(seed)

    if device == "CPU":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8"

    # Create logic CPUs
    physical_devices = tf.config.list_physical_devices("CPU")
    tf.config.experimental.set_virtual_device_configuration(
        physical_devices[0],
        [tf.config.experimental.VirtualDeviceConfiguration(),
         tf.config.experimental.VirtualDeviceConfiguration(),
         tf.config.experimental.VirtualDeviceConfiguration(),
         tf.config.experimental.VirtualDeviceConfiguration(),
         tf.config.experimental.VirtualDeviceConfiguration(),
         tf.config.experimental.VirtualDeviceConfiguration(),
         tf.config.experimental.VirtualDeviceConfiguration(),
         tf.config.experimental.VirtualDeviceConfiguration(),
         ])

    # Load list of inconsistencies from csv
    with open(incons_csv, 'r') as f:
        inconsistencies = f.readlines()
        print(f"Loaded {len(inconsistencies)} inconsistencies")

    model_id = -1
    for line in inconsistencies[1:]:
        # For each (model, input), only get the result from non-distributed CPU setting once
        flag = False
        if model_id == int(line.split(',')[0]) and input_id == int(line.split(',')[1]):
            flag = True
        model_id, input_id, setting, _ = line.strip().split(',')

        model_id = int(model_id)
        input_id = int(input_id)

        # Can only get layer results for sequential models
        if model_id < NUM_MODELS//2:
            continue

        # Non distributed
        if device == "CPU" and not flag:
            print(model_id, input_id, "0CPU")
            layer_output(model_id, input_id, "0CPU")

        # Inconsistent settings
        if device == "CPU" and "CPU" in setting:
            print(model_id, input_id, setting)
            layer_output(model_id, input_id, setting)
        if device == "GPU" and "GPU" in setting:
            print(model_id, input_id, setting)
            layer_output(model_id, input_id, setting)
