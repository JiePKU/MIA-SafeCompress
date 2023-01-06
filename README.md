## "Safety and Performance, Why not Both? Bi-Objective Optimized Model Compression toward AI Software Deployment" has been accepted by ASE2022 !!!

### SafeCompress Framework
![SafeCompress](https://github.com/JiePKU/MIA-SafeCompress/blob/master/img/SafeCompress.JPG "SafeCompress") 

### MIA-SafeCompress
This code is an instance of our *SafeCompress* framework called *MIA-SafeCompress* against membership inference attacks (MIAs)

### Requirements
* Python 3.7
* pytorch 1.8
* cuda 10.2
* datetime
* numpy
* torchvision
* itertools

### Quick Start
We take vgg with sparsity=0.05 on CIFAR100 for example
You can obtain the task acuracy (Task Acc) for vgg by running the following command:
```python
python main.py --sparse --seed 18 --sparse_init ERK  --multiplier 1 --lr 0.1 --density 0.05 --update_frequency 4000 --epochs 300 --model vgg-c --data cifar100 --decay_frequency 30000 --batch-size 128 --n_class 100
```
To perform membership inference attacks (to obtain MIA Acc), you can run:
```python
python mia_main.py --density 0.05 --epochs 100 --model vgg-c --data cifar100 --batch-size 128 --n_class 100
```

if it is helpful, please cite our paper:
```python
@inproceedings{zhu2022safety,
author = {Zhu, Jie and Wang, Leye and Han, Xiao},
title = {Safety and Performance, Why Not Both? Bi-Objective Optimized Model Compression toward AI Software Deployment},
year = {2023},
isbn = {9781450394758},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3551349.3556906},
doi = {10.1145/3551349.3556906},
booktitle = {37th IEEE/ACM International Conference on Automated Software Engineering},
articleno = {88},
numpages = {13},
keywords = {membership inference attack, test-driven development, AI software safe compression},
location = {Rochester, MI, USA},
series = {ASE22}
}
```

### Acknowledgement
This code is based on [ITOP](https://github.com/Shiweiliuiiiiiii/In-Time-Over-Parameterization)

