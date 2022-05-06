import torch
import torch.nn.functional as F
from log import  print_and_log
import numpy as np
import random

def evaluate(args, model, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    n = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.fp16: data = data.half()
            model.t = target
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print_and_log('\n{}: Classification average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Model Test evaluation' if is_test_set else 'Model Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))
    return correct / float(n)


def mia_evaluate(args, model, adversary, device, infset_loader,is_test_set=False):

    model.eval()
    adversary.eval()
    correct = 0
    n = 0
    gain = 0
    for batch_idx, ((tr_input, tr_target), (te_input, te_target)) in  infset_loader:
        # measure data loading time

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

        correct += ((member_output > 0.5) == v_is_member_labels).sum().item()
        n += member_output.size()[0]
        gain += ((v_is_member_labels==1)*(member_output-0.5)).sum() + ((0.5-member_output)*(v_is_member_labels==0)).sum()

    print_and_log('\n{}: MIA accuracy: {}/{} ({:.3f}%) MIA Gain: {:.3f}%\n'.format(
        'MIA Test evaluation' if is_test_set else 'MIA Evaluation',
        correct, n, 100. * correct / float(n), 100. *gain/float(n)))

    return correct / float(n)
