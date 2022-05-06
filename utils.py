
## this script is created for attack test before the model update

from train import mia_train
from eval import  mia_evaluate,evaluate
from MIA.model import Adversary
from log import print_and_log
import torch.optim as optim
from sparselearning.utils import get_mnist_dataloaders, get_cifar10_dataloaders, get_cifar100_dataloaders
from sparselearning.models import AlexNet, VGG16, LeNet_300_100, LeNet_5_Caffe, WideResNet, MLP_CIFAR10, ResNet34, ResNet18, DenseNet

def Attack_Test(model,args,device='cuda'):

    print_and_log("Attack training starts")
    adversary = Adversary(args.n_class).to(device)
    optimizer_mia = optim.Adam(adversary.parameters(), lr=0.001)

    if args.data == 'mnist':
        train_loader, valid_loader, test_loader, known_loader, infset_loader, refer_loader, test_infset_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)
    elif args.data == 'cifar10':
        train_loader, valid_loader, test_loader, known_loader, infset_loader, refer_loader, test_infset_loader = get_cifar10_dataloaders(args, args.valid_split,
                                                                          max_threads=args.max_threads)
    elif args.data == 'cifar100':
        train_loader, valid_loader, test_loader, known_loader, infset_loader, refer_loader, test_infset_loader = get_cifar100_dataloaders(
            args, args.valid_split, max_threads=args.max_threads)
    elif args.data == 'tinyimagenet':
        train_loader, valid_loader, test_loader, known_loader, infset_loader, refer_loader, test_infset_loader = get_cifar100_dataloaders(
            args, args.valid_split)

    size = len(refer_loader)

    for epoch in range(100):
        print_and_log("*" * 40)
        print_and_log("current epoch is {}/{}".format(epoch,100))
        print_and_log("*" * 40)

        train_private_enum = enumerate(zip(known_loader, refer_loader))
        privacy_loss, privacy_acc = mia_train(args, model, adversary, device, \
                                              train_private_enum, optimizer_mia,size=size, minmax=args.minmax)

        test_private_enum = enumerate(zip(infset_loader,test_infset_loader))
        inf_acc = mia_evaluate(args, model, adversary, device, test_private_enum)