import random
import json
from model_generation import *
from constants import *


if __name__ == "__main__":
    # For deterministic genreation, uncomment the following line
    # random.seed(0)

    # DLRM models
    for i in range(NUM_MODELS//2):
        seed = random.randint(0, 1e8)
        print(f"Generating DLRM model {i} with seed {seed}")
        generate_DLRM_model(idx=i, seed=seed)

    # Muffin models
    # subprocess.call(["bash", "./generate_muffin.sh"])
    for i in range(NUM_MODELS//2, NUM_MODELS):
        seed = random.randint(0, 1e8)
        print(f"Generating Sequential model {i} with seed {seed}")
        generate_Sequential_model(idx=i, seed=seed)
