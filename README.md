# Real-scene Reflection Removal with RAW-RGB Image Pairs
(Updating)

An official implementation code for paper "Real-scene Reflection Removal with RAW-RGB Image Pairs"

## Introduction

## Requisites

* Pytorch 1.6.0
* Python 3
* Linux

## Test

### Prepare Test Data

Download and unzip the [test set](https://drive.google.com/file/d/1pnSjX1te9DrVMotnnL8X3iwJGYb8Fjw1/view?usp=sharing), and then copy them to `datasets`.

### Download Pre-trained Model

Download and unzip our [pre-trained model](https://drive.google.com/file/d/1mCQbBi35sM9hMOxA1pjDrWvWfM-hf7Ya/view?usp=sharing), and then copy them to `checkpoints/RAW_RGB_RR`.

### Run

You can run `bash test.sh`
or equivalently:
```python
python test.py --dataroot datasets --name RAW_RGB_RR --model RAWRR --dataset_mode rawrr  --preprocess "" --no_flip --epoch final --gpu_ids 0
```

## Acknowledgement

Our code is based on [IBCLN](https://github.com/JHL-HUST/IBCLN).
