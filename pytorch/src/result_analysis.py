import pickle
import os
import sys
import traceback
import torch
import numpy as np
import csv
from config import DistributedConfig, distributed_settings
from constants import BATCH_SIZE, NUM_MODELS, NUM_TRAINING_DATASETS

def normalize_pred(pred):
    pred = torch.nn.functional.softmax(pred, -1)
    return pred

def construct_unit_test_str(setting: DistributedConfig):
    quantization = setting.quantization
    sharder_type = setting.sharder_type
    sharding_type = setting.sharding_type
    kernel_type = setting.kernel_type

    backend = setting.backend
    world_size = setting.world_size

    test_case_name = str(sharder_type) + "#" + str(sharding_type) + "#" + backend + "#" + str(world_size) + "#" + str(quantization)
    return test_case_name

path_main = "/results/outputs"

unit_tests = []
for setting in distributed_settings:
    unit_tests.append(construct_unit_test_str(setting))
print(len(unit_tests))
    
max_diff = 0
nb_compare = 0
nb_inco = 0
nb_seed = 0
nb_crash = 0

crash_list = []
incon_list = []

for seed in range(NUM_MODELS):
    nb_seed += 1
    for input_index in range(NUM_TRAINING_DATASETS):
        result_dict = {}
        standard_unit_test_not_quant = None
        standard_unit_test_quant = None
        for unit_test in unit_tests:
            new_unit_test = "output_" + str(seed) + "_" + str(input_index)+ "_" + unit_test
            unit_test_dir = os.path.join(path_main, "output_" + str(seed), "output_" + str(seed) + "_" + str(input_index), new_unit_test)
            ranks = ["rank_0.p"]
            if len(ranks) == 1:
                try:
                    rank_file = os.path.join(unit_test_dir, ranks[0])
                    with open(os.path.join(unit_test_dir, ranks[0]), "rb") as f:
                        result_dict[new_unit_test] = pickle.load(f)
                    if standard_unit_test_not_quant is None and unit_test.endswith("False"):
                        standard_unit_test_not_quant = new_unit_test
                    if standard_unit_test_quant is None and unit_test.endswith("True"):
                        standard_unit_test_quant = new_unit_test
                except KeyboardInterrupt:
                    sys.exit(1)
                except:
                    print(traceback.format_exc())
                    print("Result loading failed for ", seed, input_index, " ", new_unit_test)
            else:
                # TODO: handle multiple ranks
                pass
        print("length of result dict: ", len(result_dict))
        if len(result_dict) >= 2:
            for key, item in result_dict.items():
                nb_compare += 1

                if key.endswith("False"):
                    standard_unit_test = standard_unit_test_not_quant
                else:
                    standard_unit_test = standard_unit_test_quant

                if seed >= int(NUM_MODELS/2):
                    value_1_np = normalize_pred(result_dict[standard_unit_test][0]).detach().cpu().numpy()
                    value_2_np = normalize_pred(item[1]).detach().cpu().numpy()
                else:
                    value_1_np = result_dict[standard_unit_test][0].detach().cpu().numpy()
                    value_2_np = item[1].detach().cpu().numpy()
                if not np.allclose(value_1_np, value_2_np, atol=5e-4, rtol=1e-4):
                    nb_inco += 1
                    Linf_diff = np.max(np.abs(value_1_np - value_2_np))
                    
                    if Linf_diff > max_diff:
                        max_diff = Linf_diff
                    incon_list.append(["output_" + str(seed), "output_" + str(seed) + "_" + str(input_index), standard_unit_test, key, Linf_diff])

print("Total inconsistencies:", nb_inco)

csv_file_path = os.path.join("/results", "csv")
os.makedirs(csv_file_path, exist_ok=True)
with open(os.path.join(csv_file_path, "inconsistency.csv"), 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["model", "input", "config 1", "config 2", "Linf"])
    writer.writerows(incon_list)

