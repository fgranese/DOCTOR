# DOCTOR
DOCTOR aims to identify whether the prediction of a classifier should or should not be trusted 
so that to choose between accepting or rejecting the prediction.
### Table of the results
The results in the tables below reported in terms of AUROC% / FRR% (95% TRR).

1- Totally-Black-Box (TBB)

| Dataset |D_alpha | D_beta | SR | MHLNB | 
|------|:---------:|:--------:|:--------:|:----------:|
|[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)| <b>94 / 17.9</b> | 68.5 / 18.6 | 93.8 / 18.2 | 92.2 / 30.8 | 
|[CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)| <b>87</b> / 40.6 | 84.2 / 40.6 | 86.9 / <b>40.5</b> | 82.6 / 66.7 |
|[TinyImageNet](https://www.kaggle.com/c/thu-deep-learning/data)| <b>84.9 / 45.8</b> | <b>84.9 / 45.8</b> | <b>84.9 / 45.8</b> | 78.4 / 82.3 |
|[SVHN](http://ufldl.stanford.edu/housenumbers/)| <b>92.3 / 38.6</b> | 92.2 / 39.7 | <b>92.3 / 38.6</b> | 87.3 / 85.8 |
|[Amazon_Fashion](https://nijianmo.github.io/amazon/index.html)| <b>89.7</b> / 27.1 | <b>89.7 / 26.3 </b> | 87.4 / 50.1 | - / - |
|[Amazon_Software](https://nijianmo.github.io/amazon/index.html)| <b>68.8 / 73.2</b> | <b>68.8 / 73.2</b> | 67.3 / 86.6 | - / - |
|[IMDb](https://www.imdb.com/interfaces/)| <b>84.4 / 54.2</b> | <b>84.4</b> / 54.4 | 83.7 / 61.7 | - / - |
2- Partially-Black-Box (PBB)

| Dataset |D_alpha | D_beta | ODIN | MHLNB | 
|------|:---------:|:--------:|:--------:|:----------:|
|[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)| <b>95.2</b> / 13.9 | 94.8 / <b>13.4</b> | 94.2 / 18.4 | 84.4 / 44.6 | 
|[CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)| <b>88.2 / 35.7</b> | 87.4 / 36.7 | 87.1 / 40.7 | 50 / 94 |
|[TinyImageNet](https://www.kaggle.com/c/thu-deep-learning/data)| <b>86.1 / 43.3</b> | 85.3 / 45.1 | 84.9 / 45.3 | 59 / 86 |
|[SVHN](http://ufldl.stanford.edu/housenumbers/)| <b>93 / 36.6</b> | 92.8 / 38.4 | 92.3 / 40.7 | 88 / 54.7 |
### Current package structure
```
Package
├── data
├── datasets
├── lib_discriminators
│   ├── discriminators.py
├── models
│   └── sigmoid_nn.py
├── mystat
│   └── statistics.py
├── plots
├── tests
│   ├── compute_FRR_vs_TRR.py
│   └──  test_FRR_vs_TRR.py
├── utils
│   ├── GUI_tools.py
│   ├── dataset_utils.py
│   ├── files_utils.py
│   ├── var_utils.py
│   └── plot_utils.py
├── main.py
├── test_wrapper.py
├── README.md
└── requirements.txt

```
#### Parameter Setting
- T_tbb temperature scaling in TBB (same for SR)
- eps_tbb: perturbation magnitude in TBB (same for SR)
- T_alpha: temperature scaling in PBB for D_alpha
- eps_alpha: perturbation magnitude in PBB for D_alpha 
- T_beta: temperature scaling in PBB for D_beta
- eps_beta: perturbation magnitude in PBB for D_beta 
- T_odin: temperature scaling in PBB for ODIN
- eps_odin: perturbation magnitude in PBB for ODIN
- T_mhlnb: temperature scaling in PBB for Mahalanobis 
- eps_mhlnb: perturbation magnitude in PBB for Mahalanobis  

| Name | T_tbb | eps_tbb | T_alpha | eps_alpha | T_beta | eps_beta | T_odin | eps_odin | T_mhlnb | eps_mhlnb |
|------|:---------:|:--------:|:--------:|:----------:|:--------:|:----------:|:--------:|:----------:|:--------:|:----------:|
|[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)| 1 | 0 | 1 | 0.00035 | 1.5 | 0.00035 | 1.3 | 0 | 1 | 0.0002
|[CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)| 1 | 0 | 1 | 0.00035 | 1.5 | 0.00035 | 1.3 | 0 | 1 | 0.0002
|[TinyImageNet](https://www.kaggle.com/c/thu-deep-learning/data)| 1 | 0 | 1 | 0.00035 | 1.5 | 0.00035 | 1.3 | 0 | 1 | 0.0002
|[SVHN](http://ufldl.stanford.edu/housenumbers/)| 1 | 0 | 1 | 0.00035 | 1.5 | 0.00035 | 1.3 | 0 | 1 | 0.0002
|[Amazon_Fashion](https://nijianmo.github.io/amazon/index.html)| 1 | 0 | 1 | 0.00035 | 1.5 | 0.00035 | 1.3 | 0 | 1 | 0.0002
|[Amazon_Software](https://nijianmo.github.io/amazon/index.html)| 1 | 0 | 1 | 0.00035 | 1.5 | 0.00035 | 1.3 | 0 | 1 | 0.0002
|[IMDb](https://www.imdb.com/interfaces/)| 1 | 0 | 1 | 0.00035 | 1.5 | 0.00035 | 1.3 | 0 | 1 | 0.0002

#### Dataframe
DOCTOR requires the predictions for a given dataset to be in the following format.
Example on CIFAR10:
- 1,...,10: softmax probability associated to the corresponding class
- label: predicted class
- true_label: true class

| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | label | true_label |
|---|---|---|---|---|---|---|---|---|----|:-------:|:------------:|
|0.02|0.01|0.04|0.01|0.005|0.005|0.9|0.006|0.002|0.002|7|7|

Dataframe are stored in the corresponding directory.
For CIFAR10:
```
├── data
│   ├── cifar10_T_1_eps_0_test.csv
│   ├── cifar10_T_1_eps_0_train.csv
│   └── cifar10_T_1_eps_0_train_logits.csv
├── data_perturb
│   └── cifar10_T_1.3_eps_0_pt_odin_test.csv
├── data_perturb_our
│   ├── cifar10_T_1.5_eps_0.00035_pt_beta_test.csv
│   ├── cifar10_T_1_eps_0.0002_pt_mahalanobis_test_logits.csv
│   └── cifar10_T_1_eps_0.00035_pt_alpha_test.csv

```
#### Usage
A clean execution of DOCTOR is in:
```console
tests/test_FRR_vs_TRR.py
```

To execute it:
- Create the enviroment for DOCTOR:
```console
foo@bar:~$ conda create --name doctor python=3.8
```
- Activate the enviroment for DOCTOR:
```console
foo@bar:~$ source activate doctor
```
- Install all the required packages:
```console
(doctor) foo@bar:~$ pip install -r requirements.txt
```
- Launch the test from CLI for CIFAR10:
```console
(doctor) foo@bar:~$ python main.py -d_name cifar10 -sc tbb 
(doctor) foo@bar:~$ python main.py -d_name cifar10 -sc pbb 
(doctor) foo@bar:~$ python main.py -d_name cifar10 -sc tbb 
(doctor) foo@bar:~$ python main.py -d_name cifar10 -sc pbb 
```
Output:
```console
(doctor) foo@bar:~$ python main.py -d_name cifar10 -sc pbb -ood 
ALPHA: AUROC 95.2 % --- FRR (95% TRR) 13.9 %
BETA: AUROC 94.8 % --- FRR (95% TRR) 13.4 %
ODIN: AUROC 94.2 % --- FRR (95% TRR) 18.4 %
MAHALANOBIS: AUROC 84.4 % --- FRR (95% TRR) 44.6 %
```
Plot:
<p align="center">
<img src="https://github.com/doctor-public-submission/DOCTOR/blob/main/DOCTOR.png?raw=true" width="500"/>
</p>

Experiments with OOD samples: 
```console
(doctor) foo@bar:~$ python main.py -d_name isun_cifar10 -sc pbb -ood True
ALPHA: AUROC 95.6 % / 0.1 % --- FRR 15.1 % / 0.1 %
BETA: AUROC 95.6 % / 0.0 % --- FRR 13.6 % / 0.5 %
ODIN: AUROC 95.4 % / 0.0 % --- FRR 16.1 % / 0.2 %
ODIN (DEFAULT SETTING OF ODIN) : AUROC 93.5 % / 0.0 % --- FRR 30.6 % / 0.4 %
```
Note that, the name of the dataset to set is <code>out-dataset-name_in-dataset-name.csv</code>.

Click [here](https://drive.google.com/file/d/1v0h_1-22rGmqm1HBb-NK2e1zF9QXhc4P/view?usp=sharing) to download the datasets for OOD experiments.

#### Enviroment
We run each experiment on a machine equipped with an Intel(R) Xeon(R) 
CPU E5-2623 v4, 2.60GHz clock frequency, and a GeForce GTX 1080 Ti GPU.

We test this clean execution on a machine equipped with Intel(R) Core(TM) i7-8569U 
CPU @ 2.80GHz.



