# MIA-SafeCompress
This code is an instance of our *SafeCompress* framework called *MIA-SafeCompress* against membership inference attacks (MIAs)

## Requirements
* Python 3.7
* pytorch 1.8
* cuda 10.2
* datetime
* numpy
* torchvision
* itertools

## Quick Start


We take vgg with sparsity=0.05 on CIFAR100 for example


You can obtain the task acuracy (Task Acc) for vgg by running following command:
```python
python main.py --sparse --seed 18 --sparse_init ERK  --multiplier 1 --lr 0.1 --density 0.05 --update_frequency 4000 --epochs 300 --model vgg-c --data cifar100 --decay_frequency 30000 --batch-size 128 --n_class 100
```
We conduct this experiment on NVIDIA 1080Ti using about 4 hours. 

You can find the Task Acc in the final output in the following form
```python
Testing model and adversity
Model Test evaluation: Classification average loss: XXX, Accuracy: XXXX/XXXXXX (XXXXX)
```
The **Accuracy** is our Task Acc metric in our paper.


To perform membership inference attacks (to obtain MIA Acc), you can run:
```python
python mia_main.py --density 0.05 --epochs 100 --model vgg-c --data cifar100 --batch-size 128 --n_class 100
```
Also, you can find the MIA Acc in the final output in the follow form
```python
MIA Evaluation: MIA accuracy: XXXX/XXXXX (XXXX) MIA Gain: XXXX
```
The **MIA accuracy** is our MIA Acc metric in our paper.


