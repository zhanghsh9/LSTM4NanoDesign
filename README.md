# LSTM4NanoDesign

This is the git repo of Surface plasmon light field control based on machine learning.

## Environment
```text
python = 3.11
CUDA = 12.1
pytorch = 2.3.1
scipy = 1.10.1
numpy = 1.24.3
matplotlib = 3.8.4
```
Running on NVIDIA RTX 4070 Ti Super

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

Trained model, 150 epochs:

```text
百度网盘：
url: https://pan.baidu.com/s/1xf_w-FSIplY6Z9ddWnZLIQ
password: x5gn 
```