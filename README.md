# LSTM4NanoDesign

This is the git repo of Surface plasmon light field control based on machine learning.

## Environment
```text
python = 3.10
CUDA = 11.7
pytorch = 2.0.0
scipy = 1.10.1
numpy = 1.23.5
matplotlib = 3.7.1
```
Running on NVIDIA RTX 3080

## How to run
Train-Test split:

Use MATLAB to run split_train_test_all.m

Please notice that there are 86 step-datas in rods=2, which will not be included in the test dataset.

Data:
```text
data/
    comb_2_****_test.mat
    comb_2_****_train.mat
    comb_3_****_test.mat
    comb_3_****_train.mat
    comb_8_****_test.mat
    comb_8_****_train.mat
```

Train:
```commandline
python main.py
```

Evaluation:
```commandline
python eval.py
```

or

```commandline
python eval_attn.py
```