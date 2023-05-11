import os
import csv
import json
import pickle
import numpy as np
import argparse
import tensorflow as tf
from typing import *
from constants import *
from tensorflow import Tensor


def calc_MAD(y: Tensor, o: Tensor) -> float:
    # y: predicted value
    # o: ground truth
    return np.sum(np.abs(y - o)) / len(y)


def calc_RL(m: float, m_p: float) -> float:
    # m: mad of current layer
    # m_p: maximum mad up to current layer
    return (m - m_p) / (m_p + 1e-7)


def analyze_layer_results(m: int, i: int, s1: str, s2: str):
    # m: model idx, i: input idx, s1: setting1, s2: setting2
    result_path = f"{cur_dir}/../results/analysis/output_{m}_{i}"
    model_path = f"{cur_dir}/../data/models/model_{m}"

    layer = 0
    diff_lst = []
    mad_lst = []
    RL_lst = []

    with open(f"{result_path}/output_{s1}.pk", 'rb') as f:
        preds_1 = pickle.load(f)

    with open(f"{result_path}/output_{s2}.pk", 'rb') as f:
        preds_2 = pickle.load(f)

    for layer in range(len(preds_1)):
        pred_1 = tf.nn.softmax(preds_1[layer])
        pred_2 = tf.nn.softmax(preds_2[layer])

        # Calculate difference
        max_diff = np.max(np.abs(pred_1 - pred_2))
        diff_lst.append(max_diff)

        # Calculate MAD, assume the first setting is the ground truth
        mad = calc_MAD(pred_2, pred_1)

        # Get the model structure
        with open(f"{model_path}/model.json", 'r') as f:
            model_structure = json.load(f)["model_structure"]

        pre_layers = model_structure[str(layer)]["pre_layers"]
        if len(pre_layers) == 0:
            RL_lst.append(calc_RL(mad, 0))
        else:
            max_mad = max(mad_lst[i] for i in pre_layers)
            RL_lst.append(calc_RL(mad, max_mad))

        mad_lst.append(mad)

    print("Linf:", diff_lst)
    print("MAD:", mad_lst)
    print("RL:", RL_lst)

    idx_max_RL = np.argmax(RL_lst)
    idx_second_RL = np.argsort(RL_lst)[-2]
    print(
        f"Max RL {RL_lst[idx_max_RL]} at layer {idx_max_RL}: {model_structure[str(idx_max_RL)]['type']}")
    print(
        f"Second largest RL {RL_lst[idx_second_RL]} at layer {idx_second_RL}: {model_structure[str(idx_second_RL)]['type']}")
    print()

    row = {"Model_id": m,
           "Input_id": i,
           "Setting_1": s1,
           "Setting_2": s2,
           "Linf_output": diff_lst[-1],

           "Idx_max_RL": idx_max_RL,
           "Layer_1": model_structure[str(idx_max_RL)]['type'],
           "RL_1": RL_lst[idx_max_RL],
           "MAD_1": mad_lst[idx_max_RL],
           "Linf_1": diff_lst[idx_max_RL],

           "Idx_second_RL": idx_second_RL,
           "Layer_2": model_structure[str(idx_second_RL)]['type'],
           "RL_2": RL_lst[idx_second_RL],
           "MAD_2": mad_lst[idx_second_RL],
           "Linf_2": diff_lst[idx_second_RL]
           }

    return row


if __name__ == "__main__":
    cur_dir = os.getcwd()
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, default="../results/csv/inconsistency.csv")

    args = parser.parse_args()
    config = vars(args)

    incons_csv = config["infile"]

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    csv_header = ["Model_id", "Input_id", "Setting_1", "Setting_2", "Linf_output", 
                  "Idx_max_RL", "Layer_1", "RL_1", "MAD_1", "Linf_1",
                  "Idx_second_RL", "Layer_2", "RL_2", "MAD_2", "Linf_2"]
    rows = []

    with open(incons_csv, 'r') as f:
        inconsistencies = f.readlines()
        print(f"Loaded {len(inconsistencies)} inconsistencies")

    model_id = -1
    for line in inconsistencies[1:]:
        model_id, input_id, setting, Linf = line.strip().split(',')
        model_id = int(model_id)
        input_id = int(input_id)
        if model_id < NUM_MODELS // 2:
            continue

        print(model_id, input_id, setting)

        quantized = False
        s1 = "0CPU_q" if quantized else "0CPU"
        s2 = setting

        try:
            print(f"Setting: {s1} vs {s2}")
            rows.append(analyze_layer_results(model_id, input_id, s1, s2))

        except Exception as e:
            print(str(e))
            break

    with open(f"{cur_dir}/../results/csv/cluster.csv", 'w') as f:
        writer = csv.DictWriter(f, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(rows)
