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

You can obtain a well-trained sparse vgg by running:
```python
python main.py --sparse --seed 18 --sparse_init ERK  --multiplier 1 --lr 0.1 --density 0.05 --update_frequency 4000 --epochs 300 --model vgg-c --data cifar100 --decay_frequency 30000 --batch-size 128 --n_class 100
```

To perform membership inference attacks, you can run:
```python
file_path = "/path/to/your/checkpoints/"
python mia_main.py --density 0.05 --epochs 100 --model vgg-c --data cifar100 --batch-size 128 --n_class 100
```




