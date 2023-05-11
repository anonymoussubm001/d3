# D3 TensorFlow

This is the repo for reproducing the TensorFlow experiment in D3: Differential Testing of Distributed Deep Learning with Model Generation

## 1. Setup Muffin docker

```shell
docker pull librarytesting/muffin:E1
docker run --runtime=nvidia -it -v $PWD/muffin:/data --name muffin librarytesting/muffin:E1 /bin/bash
```

(now inside muffin docker)
```shell
source activate lemon
cd /data/dataset
python get_dataset.py cifar10
```

Now you can exit the Muffin docker.

## 2. Setup TensorFlow docker

```shell
docker pull tensorflow/tensorflow:devel-gpu
docker run -it --name D3-tf --gpus all -v "$PWD":"/mnt" -v "$PWD/../data":"/data" -v "$PWD/../results":"/results" -w "/mnt" tensorflow/tensorflow:devel-gpu bash
```

(now inside TensorFlow docker)
```shell
pip install tensorflow==2.11.0 tensorflow_addons==0.19.0 tensorflow-model-optimization==0.7.3
```

Now you can exit the TensorFlow docker.

NOTE: After the one-time setup, use the following command to enter docker environment in the future to prevent permission issue when using the docker:
```shell
docker exec -u $(id -u):$(id -g) -it D3-tf bash
```

## 3. Setup folders

```shell
chmod +x ./setup.sh && ./setup.sh
```

## 4. Generate Muffin models

```shell
./generate_muffin.sh
```

NOTE: If you encounter permission issues, do
```shell
chmod 777 ./muffin/data
```

## **All the following commands should be executed inside the TensorFlow docker environment**

## 5. Generate models and inputs

You can control the batch size, number of models and number of inputs generated in `constants.py`. 
The default values in this demo are: `BATCH_SIZE=2400, NUM_MODELS=4, NUM_TRAINING_DATASETS=2`.
To reproduce the results in the paper, use `BATCH_SIZE=2400, NUM_MODELS=200, NUM_TRAINING_DATASETS=10`.

```shell
python3 ./rename_muffin.py
python3 ./convert_to_quantized.py
python3 ./gen_model_and_input.py
```

## 6. Run the experiment

You can change the constants in `run_multiple_process.py` to run less settings. Default behavior is to run all settings.

```shell
python3 ./run_multiple_process.py
```

## 7. Result analysis
Generate a csv file including all the inconsistencies
```shell
python3 ./result_analysis.py [--quantized]
```
