# LSTM4NanoDesign

This is the git repo of Surface plasmon light field control based on machine learning.

## Environment
```text
python = 3.12.4
CUDA = 11.8
pytorch = 2.3.1
scipy = 1.14.0
numpy = 2.0.0
matplotlib = 3.8.4
```
Running on 4 NVIDIA RTX 2080 Ti

## How to run
### Train-Test split:

Use MATLAB to run split_train_test_all.m

Please notice that there are 86 step-datas in rods=2, which will not be included in the test dataset.

### Data:
```text
data/
    comb_2_****_test.mat
    comb_2_****_train.mat
    comb_3_****_test.mat
    comb_3_****_train.mat
    comb_8_****_test.mat
    comb_8_****_train.mat
```

### Train:
#### Forward network: 
```commandline
python train_forward*.py
```

### Evaluation:
```commandline
python eval*.py
```
