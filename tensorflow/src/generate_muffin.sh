#!/bin/bash

docker exec -it muffin bash -c "cd data && source activate lemon && sqlite3 ./data/cifar10.db < ./data/create_db.sql"

python3 ./setup_muffin.py --mode seq
docker exec -it muffin bash -c "cd data && source activate lemon && CUDA_VISIBLE_DEVICES=-1 python run.py"
python3 ./setup_muffin.py --mode dag
docker exec -it muffin bash -c "cd data && source activate lemon && CUDA_VISIBLE_DEVICES=-1 python run.py"
