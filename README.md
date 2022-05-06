# MIA-SafeCompress
This code is an instance of our *SafeComrpess* framework called *MIA-SafeCompress* against membership inference attacks (MIAs)

![数据格式](https://github.com/JiePKU/MIA-SafeCompress/blob/master/img/SafeCompress.JPG "数据格式")

## Requirements
* Python 3.8
* pytorch 1.8
* cuda 11.1
* datetime
* numpy
* torchvision
* itertools

## Quick Start

You can ontain a well-trained sparse vgg on CIFAR100 by running:
```python
python main.py --sparse --seed 18 --sparse_init ERK  --multiplier 1 --lr 0.1 --density 0.05 --update_frequency 4000 --epochs 300 --model vgg-c --data cifar100 --decay_frequency 30000 --batch-size 128 --n_class 100
```

To perform membership inference attacks, you can run:
```python
file_path = "/path/to/your/checkpoints/"
python mia_main.py --density 0.05 --epochs 100 --model vgg-c --data cifar100 --batch-size 128 --n_class 100
```




