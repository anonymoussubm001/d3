import os
import csv
import pickle
import argparse
import numpy as np
import tensorflow as tf
from constants import *
from tensorflow import Tensor
from constants import *


def read_result(model: int, idx: int, rtol, atol):
    result_path = f"{cur_dir}/../results/outputs/output_{model}/"

    if not os.path.exists(result_path + f"output_{idx}_{configs[0]}.pk"):
        raise Exception(f"{configs[0]} not found")

    preds = []
    for c in configs:
        with open(result_path + f"output_{idx}_{c}.pk", 'rb') as f:
            pred: Tensor = pickle.load(f)
            preds.append(pred)

    diffs = {}
    closes = []

    if model >= NUM_MODELS // 2:
        for i in range(len(preds)):
            preds[i] = tf.nn.softmax(preds[i])

    for i in range(1, len(preds)):
        pred_1, pred_2 = preds[0], preds[i]

        max_diff = np.max(np.abs(pred_1 - pred_2))
        diffs[configs[i]] = float(max_diff)

        closes.append(np.allclose(pred_1, pred_2, rtol=rtol, atol=atol))

    return np.array(closes), diffs


def get_result_for_input(input_idx: int):
    closes, diffs = read_result(
        model_idx, input_idx, rtol=1e-4, atol=5e-4)

    n_inconsistency = sum(1 for x in closes if x == False)

    if n_inconsistency == 0:
        return False

    # Inconsistency detected
    print(f"Model {model_idx} input {input_idx} inconsistency {n_inconsistency}, max diff: {max(diffs.values())}")
    print("Inconsistent settings: ", end="")
    for (i, c) in enumerate(closes):
        if c == False:
            inconsistency_list.append({
                "model": model_idx,
                "input": input_idx,
                "config": configs[i+1],
                "Linf": diffs[configs[i+1]]
            })
            print(configs[i+1], end=", ")
    print()

    global global_inconsistency
    global_inconsistency += n_inconsistency
    return True


def get_result_for_model(model_idx: int):
    model_inconsist = False
    for input_idx in range(NUM_TRAINING_DATASETS):
        try:
            model_inconsist |= get_result_for_input(input_idx)
        except Exception as e:
            failed_models.add(model_idx)
            print(model_idx, e)

    if model_inconsist:
        inconsist_models.append(model_idx)
        global n_inconsist_model
        n_inconsist_model += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantized", action='store_true')

    args = parser.parse_args()
    config = vars(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    cur_dir = os.getcwd()

    global_inconsistency = 0
    n_inconsist_model = 0

    # Change configs accordingly if less settings are used
    # The first one must be "0CPU" or "0CPU_q"
    if config["quantized"]:
        configs = ["0CPU_q", "1CPU_q", "2CPU_q", "3CPU_q",
                   "4CPU_q", "8CPU_q",
                   "0GPU_q", "1GPU_q", "2GPU_q", "3GPU_q",
                   "4GPU_q", "8GPU_q"]
    else:
        configs = ["0CPU", "1CPU", "2CPU", "3CPU",
                   "4CPU", "8CPU",
                   "0GPU", "1GPU", "2GPU", "3GPU",
                   "4GPU", "8GPU"]

    inconsist_models = []
    failed_models = set()
    inconsistency_list = []

    headers = ["model", "input", "config", "Linf"]

    for model_idx in range(NUM_MODELS):
        get_result_for_model(model_idx)

    # print("Failed models:", len(failed_models))
    # print(failed_models)
    # print("Model with inconsistencies:", n_inconsist_model)
    # print(inconsist_models)
    print("Total inconsistencies:", global_inconsistency)

    if config["quantized"]:
        file_path = f"{cur_dir}/../results/csv/inconsistency_q.csv"
    else:
        file_path = f"{cur_dir}/../results/csv/inconsistency.csv"
    with open(file_path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(inconsistency_list)
