
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from itertools import cycle
from log import  print_and_log
import time
from MIA.entropy_regularization import EntropyLoss,ThresholdEntropyLoss,KLEntropyLoss,AguEntropyLoss, L2_Re


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class CELoss(nn.Module):
    def __init__(self):
        super(CELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self,pred,label):
        loss = self.ce(pred,label)
        return loss,torch.Tensor([0])

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def KD_loss(t_out,s_out):
    t_out = torch.softmax(t_out,dim=1)
    s_out = torch.log_softmax(s_out,dim=1)
    loss = (-t_out*s_out).sum(dim=1).mean()
    return loss

def kd_train(args, model, teacher, device, train_enum, optimizer, size, mask=None, num_batches=10000):

    model.train()
    teacher.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    entroys = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    first_id = -1

    if args.regularization != None:
        if args.regularization == "EntropyLoss":
            criterion = EntropyLoss()
        elif args.regularization == "ThresholdEntropyLoss":
            criterion = ThresholdEntropyLoss()
        elif args.regularization == "KLEntropyLoss":
            criterion = KLEntropyLoss(n_class=args.n_class)
        else:
            criterion = AguEntropyLoss(beta=0.08)
    else:
        criterion = CELoss()

    for batch_idx, (data, target) in train_enum:

        if first_id == -1:
            first_id = batch_idx

        data, target = data.to(device), target.to(device)
        data_time.update(time.time() - end)

        if args.fp16: data = data.half()
        with torch.no_grad():
            t_out = teacher(data)
        optimizer.zero_grad()
        output = model(data)

        loss1, entroy = criterion(output, target)
        loss =  (loss1 + KD_loss(t_out,output))/2
        # entroy = L2_Re(model,1e-4)
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        entroys.update(entroy.item(),data.size(0))
        losses.update(loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if mask is not None: mask.step()
        else: optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx % 100 == 0:
            print_and_log(
                '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | Entroy: {entroy:.4f} |top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=size,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    entroy = entroys.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if batch_idx - first_id >= num_batches:
            break
    return (losses.avg, top1.avg)

def train(args, model, inference_model, device, train_enum, optimizer, size, mask=None, num_batches=10000):

    model.train()
    inference_model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    entroys = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    first_id = -1

    if args.regularization != None:
        if args.regularization == "EntropyLoss":
            criterion = EntropyLoss()
        elif args.regularization == "ThresholdEntropyLoss":
            criterion = ThresholdEntropyLoss()
        elif args.regularization == "KLEntropyLoss":
            criterion = KLEntropyLoss(n_class=args.n_class)
        else:
            criterion = AguEntropyLoss(beta=0.08)
    else:
        criterion = CELoss()

    for batch_idx, (data, target) in train_enum:

        if first_id == -1:
            first_id = batch_idx

        data, target = data.to(device), target.to(device)
        data_time.update(time.time() - end)

        if args.fp16: data = data.half()
        optimizer.zero_grad()
        output = model(data)

        loss, entroy = criterion(output, target)
        # entroy = L2_Re(model,1e-4)
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        entroys.update(entroy.item(),data.size(0))
        losses.update(loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))

        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if mask is not None: mask.step()
        else: optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx % 100 == 0:
            print_and_log(
                '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | Entroy: {entroy:.4f} |top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=size,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    entroy = entroys.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if batch_idx - first_id >= num_batches:
            break
    return (losses.avg, top1.avg)

def mia_train(args, model, adversary, device,\
                      train_private_enum, optimizer_mia, size ,minmax = False, num_batchs=1000):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    adversary.train()
    model.eval()
    # train inference model
    # from itertools import cycle
    # for batch_idx, (data, target) in enumerate(zip(known_loader, cycle(refer_loader))):
    end = time.time()
    first_id = -1
    # when short dataloader is over, thus end
    for batch_idx, ((tr_input, tr_target), (te_input, te_target)) in  train_private_enum:
        # measure data loading time

        if first_id == -1:
            first_id = batch_idx

        data_time.update(time.time() - end)
        tr_input = tr_input.to(device)
        te_input = te_input.to(device)
        tr_target = tr_target.to(device)
        te_target = te_target.to(device)
        # compute output
        model_input = torch.cat((tr_input, te_input))

        if args.fp16: model_input = model_input.half()

        with torch.no_grad():
            pred_outputs = model(model_input)

        infer_input = torch.cat((tr_target, te_target))

        one_hot_tr = torch.from_numpy((np.zeros((infer_input.size(0), args.n_class)) - 1)).cuda().type(torch.cuda.FloatTensor)
        infer_input_one_hot = one_hot_tr.scatter_(1, infer_input.type(torch.cuda.LongTensor).view([-1, 1]).data, 1)

        attack_model_input = pred_outputs  # torch.cat((pred_outputs,infer_input_one_hot),1)
        v_is_member_labels = torch.from_numpy(
            np.reshape(np.concatenate((np.ones(tr_input.size(0)), np.zeros(te_input.size(0)))), [-1, 1])).cuda().type(torch.cuda.FloatTensor)

        r = np.arange(v_is_member_labels.size()[0]).tolist()
        random.shuffle(r)
        attack_model_input = attack_model_input[r]
        v_is_member_labels = v_is_member_labels[r]
        infer_input_one_hot = infer_input_one_hot[r]

        member_output = adversary(attack_model_input, infer_input_one_hot)

        loss = F.binary_cross_entropy(member_output, v_is_member_labels)

        # measure accuracy and record loss
        prec1 = np.mean((member_output.data.cpu().numpy() > 0.5) == v_is_member_labels.data.cpu().numpy())
        losses.update(loss.item(), model_input.size(0))
        top1.update(prec1, model_input.size(0))

        # compute gradient and do SGD step
        optimizer_mia.zero_grad()
        if args.fp16:
            optimizer_mia.backward(loss)
        else:
            loss.backward()

        optimizer_mia.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx - first_id > num_batchs:
            break

        # plot progress
        if batch_idx % 10 == 0:
            print_and_log(
                '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | top1: {top1: .4f} '.format(
                    batch=batch_idx,
                    size=size,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                ))

    return (losses.avg, top1.avg)




def train_privately(args, model, inference_model, device, train_enum, optimizer, size, mask=None, num_batches=10000):
    model.train()
    inference_model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mia_losses = AverageMeter()
    entroys = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    first_id = -1

    if args.regularization != None:
        if args.regularization == "EntropyLoss":
            criterion = EntropyLoss()
        elif args.regularization == "ThresholdEntropyLoss":
            criterion = ThresholdEntropyLoss()
        elif args.regularization == "KLEntropyLoss":
            criterion = KLEntropyLoss(n_class=args.n_class)
        else:
            criterion = AguEntropyLoss(beta=0.08)
    else:
        criterion = CELoss()

    for batch_idx, (data, target) in train_enum:

        if first_id == -1:
            first_id = batch_idx

        data, target = data.to(device), target.to(device)

        data_time.update(time.time() - end)

        if args.fp16: data = data.half()
        optimizer.zero_grad()
        output = model(data)

        one_hot_label = torch.from_numpy(np.zeros((data.size()[0],args.n_class))-1).to(device).float()
        one_hot_label = one_hot_label.scatter_(1,target.type(torch.cuda.LongTensor).view([-1, 1]).data,1)

        mia_label = torch.from_numpy(np.ones(data.size()[0])).to(device).float()
        mia_out = inference_model(output,one_hot_label)

        loss1, entroy = criterion(output, target)
        mia_loss = F.binary_cross_entropy(mia_out, mia_label)
        loss = loss1 + 0.5 * mia_loss

        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        entroys.update(entroy.item(), data.size(0))
        losses.update(loss1.item(), data.size(0))
        mia_losses.update(mia_loss,data.size(0))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))

        if args.fp16: optimizer.backward(loss)
        else:
            loss.backward()

        if mask is not None:
            mask.step()
        else:
            optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if batch_idx % 100 == 0:
            print_and_log(
                '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | | Loss: {loss:.4f} | | MIA Loss: {mia_loss:.4f}  | Entroy: {entroy:.4f} |top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=size,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    mia_loss = mia_losses.avg,
                    entroy=entroys.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if batch_idx - first_id >= num_batches:
            break

    return (losses.avg, top1.avg)




