import torch
import torch.nn as nn
from MIA.model import Adversary
from sparselearning.utils import get_mnist_dataloaders, get_cifar10_dataloaders, get_cifar100_dataloaders, get_tinyimagenet_dataloaders
from sparselearning.models import AlexNet, VGG16, LeNet_300_100, LeNet_5_Caffe, WideResNet, MLP_CIFAR10, ResNet34, ResNet18, DenseNet
from train import mia_train
import torch.backends.cudnn as cudnn
import torch.optim as optim
from eval import mia_evaluate,evaluate
import datetime

def load_model(file_path,model):
    param = torch.load(file_path,map_location='cpu')
    model_param = param['model']['state_dict']
    model.load_state_dict(model_param)
    return model

def save_checkpoints(adversary,epoch,optimizer,file_name):
    obj = { 'epoch': epoch,
            'state_dict': adversary.state_dict(),
            'optimizer': optimizer.state_dict(),
    }
    torch.save(obj,file_name)



from log import  print_and_log,setup_logger

models = {}
models['MLPCIFAR10'] = (MLP_CIFAR10,[])
models['lenet5'] = (LeNet_5_Caffe,[])
models['lenet300-100'] = (LeNet_300_100,[])
models['DenseNet'] = (DenseNet,[10])
models['ResNet34'] = ()
models['ResNet18'] = ()
models['alexnet-s'] = (AlexNet, ['s', 100])
models['alexnet-b'] = (AlexNet, ['b', 100])
models['vgg-c'] = (VGG16, ['C', 100])
models['vgg-d'] = (VGG16, ['D', 10])
models['vgg-like'] = (VGG16, ['like', 10])
models['wrn-28-2'] = (WideResNet, [28, 2, 10, 0.3])
models['wrn-22-8'] = (WideResNet, [22, 8, 10, 0.3])
models['wrn-16-8'] = (WideResNet, [16, 8, 10, 0.3])
models['wrn-16-10'] = (WideResNet, [16, 10, 10, 0.3])
logger = None
cudnn.benchmark = True
cudnn.deterministic = True
import  argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')

    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 100)')
    randomhash = "-".join("-".join(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split(" ")).split(":"))
    parser.add_argument('--save', type=str, default=randomhash + '.pth',
                        help='path to save the final model')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument("--n_class", type=int, default=100, help="number of class in dataset using now")
    parser.add_argument('--density', type=float, default=0.05, help='The density of the overall sparse network.')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--datadir', type=str, default='./data/tiny-imagenet-200/')
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument("--minmax", action='store_true', help='If conbining with minmax strategy')
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')

    args = parser.parse_args()
    device = 'cpu'
    setup_logger(args)
    print_and_log(args)

    if args.model not in models:
        print('You need to select an existing model via the --model argument. Available models include: ')
        for key in models:
            print('\t{0}'.format(key))
        raise Exception('You need to select a model')
    elif args.model == 'ResNet18':
        model = ResNet18(c=200).to(device)
    elif args.model == 'ResNet34':
        model = ResNet34(c=100).to(device)
    else:
        cls, cls_args = models[args.model]
        model = cls(*(cls_args + [args.save_features, args.bench])).to(device)

    if args.data == 'mnist':
        train_loader, valid_loader, test_loader, known_loader, infset_loader, refer_loader, test_infset_loader = get_mnist_dataloaders(args, validation_split=args.valid_split)
    elif args.data == 'cifar10':
        train_loader, valid_loader, test_loader, known_loader, infset_loader, refer_loader, test_infset_loader = get_cifar10_dataloaders(args, args.valid_split,
                                                                          max_threads=args.max_threads)
    elif args.data == 'cifar100':
        train_loader, valid_loader, test_loader, known_loader, infset_loader, refer_loader, test_infset_loader = get_cifar100_dataloaders(
            args, args.valid_split, max_threads=args.max_threads)
    elif args.data == 'tinyimagenet':
        train_loader, valid_loader, test_loader, known_loader, infset_loader, refer_loader, test_infset_loader = get_tinyimagenet_dataloaders(
            args, args.valid_split)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    adversary = Adversary(args.n_class).to(device)
    print_and_log(adversary)
    optimizer_mia = optim.Adam(adversary.parameters(), lr=0.001)
    file_path = "./checkpoints/2022-05-01-23-16-27.pth"
    model = load_model(file_path,model)
    model.to(device)
    evaluate(args, model, device, test_loader, is_test_set=False)
    size = len(refer_loader)

    best_mia_acc = 0
    for epoch in range(args.epochs):
        print_and_log("*" * 40)
        print_and_log("current epoch is {}/{}".format(epoch,args.epochs))
        print_and_log("*" * 40)

        train_private_enum = enumerate(zip(known_loader, refer_loader))
        privacy_loss, privacy_acc = mia_train(args, model, adversary, device, \
                                              train_private_enum, optimizer_mia,size=size, minmax=args.minmax)

        test_private_enum = enumerate(zip(infset_loader,test_infset_loader))
        inf_acc = mia_evaluate(args, model, adversary, device, test_private_enum)

        if inf_acc > best_mia_acc:
            best_mia_acc = inf_acc
            save_checkpoints(adversary,epoch,optimizer_mia,'./mia_checkpoints/'+args.save)

    print_and_log("=" * 40)
    param = torch.load('./mia_checkpoints/'+args.save)
    adversary.load_state_dict(param['state_dict'])
    print_and_log("the best mia acc epoch:{}".format(param['epoch']))
    test_private_enum = enumerate(zip(infset_loader, test_infset_loader))
    inf_acc = mia_evaluate(args, model, adversary, device, test_private_enum,is_test_set=True)
