from __future__ import print_function

import os
import argparse
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import sparselearning
from sparselearning.core import Masking, CosineDecay, LinearDecay
from sparselearning.models import AlexNet, VGG16, LeNet_300_100, LeNet_5_Caffe, WideResNet, MLP_CIFAR10, ResNet34, ResNet18,DenseNet
from sparselearning.utils import get_mnist_dataloaders, get_cifar10_dataloaders, get_cifar100_dataloaders,get_tinyimagenet_dataloaders
import warnings
import datetime
from eval import evaluate,mia_evaluate
from train import train,mia_train, train_privately, kd_train
from MIA import Adversary, EntropyLoss, NegetiveInferenceGain
warnings.filterwarnings("ignore", category=UserWarning)
cudnn.benchmark = True
cudnn.deterministic = True

if not os.path.exists('./models'): os.mkdir('./models')
if not os.path.exists('./logs'): os.mkdir('./logs')
logger = None

models = {}
models['MLPCIFAR10'] = (MLP_CIFAR10,[])
models['lenet5'] = (LeNet_5_Caffe,[])
models['lenet300-100'] = (LeNet_300_100,[])
models['ResNet34'] = ()
models['ResNet18'] = ()
models['DenseNet'] = (DenseNet,[10])
models['alexnet-s'] = (AlexNet, ['s', 10])
models['alexnet-b'] = (AlexNet, ['b', 10])
models['vgg-c'] = (VGG16, ['C', 100])
models['vgg-d'] = (VGG16, ['D', 10])
models['vgg-like'] = (VGG16, ['like', 10])
models['wrn-28-2'] = (WideResNet, [28, 2, 10, 0.3])
models['wrn-22-8'] = (WideResNet, [22, 8, 10, 0.3])
models['wrn-16-8'] = (WideResNet, [16, 8, 10, 0.3])
models['wrn-16-10'] = (WideResNet, [16, 10, 10, 0.3])
from log import print_and_log, setup_logger


def save_checkpoints(model,epoch,optimizer,lr_scheduler,file_name, mask=None):
    assert isinstance(model, list), "model type require list"
    assert isinstance(optimizer, list), "optimizer type require list"
    obj = { 'epoch':epoch,
            'model':{'state_dict':model[0].state_dict(),
                      'optimizer':optimizer[0].state_dict(),
                     'lr_scheduler': lr_scheduler.state_dict(),
                     'mask': mask if mask is not None else None
                     },
            'MIA':{
                'state_dict': model[1].state_dict(),
                'optimizer': optimizer[1].state_dict(),
            }
    }
    torch.save(obj,file_name)

def load_checkpoints(model,adversary,file_name):
    param = torch.load(file_name)
    model.load_state_dict(param['model']['state_dict'])
    adversary.load_state_dict(param['MIA']['state_dict'])
    return model,adversary

def resume_from_file(args, model, optimizer, lr_scheduler, adversary=None, optimizer_mia=None):
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']

    ### load model
    model.load_state_dict(checkpoint['model']['state_dict'])
    optimizer.load_state_dict(checkpoint['model']['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['model']['lr_scheduler'])
    mask = checkpoint['model']['mask']

    ### load adversary
    if adversary is not None and optimizer_mia is not None:
        adversary.load_state_dict(checkpoint['MIA']['state_dict'])
        optimizer_mia.load_state_dict(checkpoint['MIA']['optimizer'])
        return model, optimizer, lr_scheduler, mask, adversary, optimizer_mia

    return model, optimizer, lr_scheduler, mask


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--multiplier', type=int, default=1, metavar='N',
                        help='extend training time by multiplier times')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=17, metavar='S', help='random seed (default: 17)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
    randomhash = "-".join("-".join(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').split(" ")).split(":"))
    parser.add_argument('--save', type=str, default=randomhash + '.pth',
                        help='path to save the final model')
    parser.add_argument('--data', type=str, default='mnist')
    parser.add_argument('--datadir', type=str, default='./data/tiny-imagenet-200/')
    parser.add_argument('--decay_frequency', type=int, default=25000)
    parser.add_argument('--l1', type=float, default=0.0)
    parser.add_argument('--fp16', action='store_true', help='Run in fp16 mode.')
    parser.add_argument('--valid_split', type=float, default=0.1)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--start-epoch', type=int, default=1)
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--l2', type=float, default=5.0e-4)
    parser.add_argument('--iters', type=int, default=1, help='How many times the model should be run after each other. Default=1')
    parser.add_argument('--save-features', action='store_true', help='Resumes a saved model and saves its feature data to disk for plotting.')
    parser.add_argument('--bench', action='store_true', help='Enables the benchmarking of layers and estimates sparse speedups')
    parser.add_argument('--max-threads', type=int, default=10, help='How many threads to use for data loading.')

    # training inference attack model strategy
    parser.add_argument("--minmax", action='store_true',help='If conbining with minmax strategy')
    parser.add_argument("--n_class",type=int, default=100,help="number of class in dataset using now")

    # pretrain classification model
    parser.add_argument("--pretrain_epoch",type=int, default=200, help="pretrain the model effeciently")

    # regularization term
    parser.add_argument("--regularization", type=str, default=None, help="EntropyLoss,AguEntropyLoss,ThresholdEntropyLoss,KLEntropyLoss")

    # ITOP settings
    sparselearning.core.add_sparse_args(parser)

    args = parser.parse_args()
    setup_logger(args)
    print_and_log(args)

    if args.fp16:
        try:
            from apex.fp16_utils import FP16_Optimizer
        except:
            print('WARNING: apex not installed, ignoring --fp16 option')
            args.fp16 = False

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print_and_log('\n\n')
    print_and_log('='*80)
    torch.manual_seed(args.seed)
    for i in range(args.iters):
        print_and_log("\nIteration start: {0}/{1}\n".format(i+1, args.iters))

        if args.data == 'mnist':
            train_loader, valid_loader, test_loader, known_loader, infset_loader, refer_loader, test_infset_loader  = get_mnist_dataloaders(args, validation_split=args.valid_split)
        elif args.data == 'cifar10':
            train_loader, valid_loader, test_loader, known_loader, infset_loader, refer_loader, test_infset_loader = get_cifar10_dataloaders(args, args.valid_split, max_threads=args.max_threads)
        elif args.data == 'cifar100':
            train_loader, valid_loader, test_loader, known_loader, infset_loader, refer_loader, test_infset_loader = get_cifar100_dataloaders(args, args.valid_split, max_threads=args.max_threads)
        elif args.data == 'tinyimagenet':
            train_loader, valid_loader, test_loader, known_loader, infset_loader, refer_loader, test_infset_loader = get_tinyimagenet_dataloaders(
                args, args.valid_split)

        if args.model not in models:
            print('You need to select an existing model via the --model argument. Available models include: ')
            for key in models:
                print('\t{0}'.format(key))
            raise Exception('You need to select a model')
        elif args.model == 'ResNet18':
            model = ResNet18(c=100).to(device)
        elif args.model == 'ResNet34':
            model = ResNet34(c=100).to(device)
        else:
            cls, cls_args = models[args.model]
            model = cls(*(cls_args + [args.save_features, args.bench])).to(device)
        print_and_log(model)
        print_and_log('=' * 60)
        print_and_log(args.model)
        print_and_log('=' * 60)
        # a  trained model
        # teacher = cls(*(cls_args + [args.save_features, args.bench]))
        # teacher.load_state_dict(torch.load("./checkpoints/vgg16_trained_model_on_cifar100.pth",map_location="cpu")["state_dict"])
        # teacher = teacher.to(device)
        # MIA model
        adversary = Adversary(args.n_class).to(device)
        print_and_log(adversary)
        optimizer_mia = optim.Adam(adversary.parameters(), lr=0.001)

        optimizer = None
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2, nesterov=True)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        else:
            print('Unknown optimizer: {0}'.format(args.optimizer))
            raise Exception('Unknown optimizer.')

        ## TODO  learning rate update for framework
        milestone = [int(args.epochs / 2) * args.multiplier, int(args.epochs * 3 / 4) * args.multiplier]

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= milestone, last_epoch=-1)

        if args.fp16:
            print('FP16')
            optimizer = FP16_Optimizer(optimizer,
                                       static_loss_scale = None,
                                       dynamic_loss_scale = True,
                                       dynamic_loss_args = {'init_scale': 2 ** 16})
            model = model.half()

        mask = None
        if args.sparse and args.resume is None:
            decay = CosineDecay(args.death_rate, len(train_loader)*(args.epochs*args.multiplier))
            mask = Masking(optimizer, death_rate=args.death_rate, death_mode=args.death, death_rate_decay=decay, growth_mode=args.growth,
                           redistribution_mode=args.redistribution, args=args)
            mask.add_module(model, sparse_init=args.sparse_init, density=args.density)

        if args.resume:
            if os.path.isfile(args.resume):
                print_and_log("=> loading checkpoint '{}'".format(args.resume))
                model, optimizer, lr_scheduler, mask_temp,adversary, optimizer_mia = \
                    resume_from_file(args, model, optimizer, lr_scheduler,adversary,optimizer_mia)
                if mask_temp is not None:
                    mask = mask_temp
                print_and_log("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, args.start_epoch))
                print_and_log('Testing...')
                evaluate(args, model, device, test_loader)
                mia_evaluate(args, model, adversary, device, enumerate(infset_loader), enumerate(test_infset_loader))
                print('*'*60)
                # model.feats = []
                # model.densities = []
                # plot_class_feature_histograms(args, model, device, train_loader, optimizer)
            else:
                print_and_log("=> no checkpoint found at '{}'".format(args.resume))

        best_acc = 0.0
        best_inf_acc = 0.0
        size = len(train_loader)
        size_r = len(refer_loader)
        time = size_r // 2

        if args.minmax:
            for epoch in range(args.start_epoch, args.epochs*args.multiplier + 1):
                print_and_log("*"*40)
                print_and_log("current epoch is {}".format(epoch))
                print_and_log("*" * 40)
                train_enum = enumerate(train_loader)
                train_private_enum = enumerate(zip(known_loader, refer_loader))
                test_private_enum = enumerate(zip(infset_loader,test_infset_loader))

                if epoch < args.pretrain_epoch:
                    _ , __ = train(args, model, adversary, device, train_enum , optimizer, size, mask)

                else:
                    for i in range(size//2):
                        privacy_loss, privacy_acc = mia_train(args, model, adversary, device, \
                                train_private_enum, optimizer_mia, size, args.minmax, num_batchs=1)
                        train_loss, train_acc = train_privately(args, model, adversary, device, train_enum, optimizer, size, mask, num_batches=1)

                        if i%10==0:
                            print_and_log('privacy res: {}'.format(privacy_acc) +'    ' + 'train acc: {}'.format(train_acc))

                        if (i+1)% time ==0:
                            train_private_enum = enumerate(zip(known_loader, refer_loader))


                lr_scheduler.step()
                if args.valid_split > 0.0:
                    val_acc = evaluate(args, model, device, valid_loader)
                    if val_acc > best_acc:
                        print('Saving model')
                        best_acc = val_acc
                        save_checkpoints([model, adversary], epoch, [optimizer, optimizer_mia], lr_scheduler,
                                         "./checkpoints/" + args.save, mask)

                    if epoch >= args.pretrain_epoch:
                        inf_acc = mia_evaluate(args, model, adversary, device, test_private_enum)
                        # if inf_acc > best_inf_acc:
                        #     print("Saving adversary")
                        #     best_inf_acc = inf_acc
                        #     save_checkpoints([model, adversary], epoch, [optimizer, optimizer_mia], lr_scheduler,
                        #                      "./checkpoints/adv-" + args.save, mask)

        else:
            for epoch in range(args.start_epoch, args.epochs * args.multiplier + 1):
                print_and_log("*" * 40)
                print_and_log("current epoch is {}".format(epoch))
                print_and_log("*" * 40)
                train_enum = enumerate(train_loader)
                _, __ = train(args, model, adversary, device, train_enum, optimizer, size, mask)
                #_, __ = kd_train(args, model, teacher, device, train_enum, optimizer, size, mask)

                lr_scheduler.step()
                if args.valid_split > 0.0:
                    val_acc = evaluate(args, model, device, valid_loader)
                    if val_acc > best_acc:
                        print('Saving model')
                        best_acc = val_acc
                        save_checkpoints([model,adversary],epoch,[optimizer,optimizer_mia],lr_scheduler,"/output/"+args.save, mask)

        print('Testing model and adversity')
        model, adversary = load_checkpoints(model,adversary,"/output/" + args.save)
        evaluate(args, model, device, test_loader, is_test_set=True)
        if args.minmax:
            test_private_enum = enumerate(zip(infset_loader,test_infset_loader))
            mia_evaluate(args, model, adversary, device, test_private_enum, is_test_set=True)
        print_and_log("\nIteration end: {0}/{1}\n".format(i+1, args.iters))

        layer_fired_weights, total_fired_weights = mask.fired_masks_update()
        for name in layer_fired_weights:
            print_and_log('The final percentage of fired weights in the layer', name, 'is:', layer_fired_weights[name])
        print_and_log('The final percentage of the total fired weights is:', total_fired_weights)


if __name__ == '__main__':
   main()
