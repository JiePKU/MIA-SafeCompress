import torch
import torch.nn as nn

"""
refer to : "REGULARIZING NEURAL NETWORKS BY PENALIZING CONFIDENT OUTPUT DISTRIBUTIONS"
"""

class EntropyLoss(nn.Module):
    def __init__(self,beta=0.1):
        super(EntropyLoss, self).__init__()
        self.beta = beta
        self.nll = nn.NLLLoss(reduction='none')

    def forward(self,prob,label):
        # prob B C
        prob = torch.log_softmax(prob, dim=1)
        entropy = - (prob.exp() * prob).sum(dim=1) # B
        return (self.nll(prob,label) - self.beta * entropy).mean(), entropy.mean()


class ThresholdEntropyLoss(nn.Module):
    def __init__(self,beta=0.1,threshold=0.5):
        super(ThresholdEntropyLoss, self).__init__()
        self.beta = beta
        self.threshold = threshold
        self.nll = nn.NLLLoss(reduction='none')

    def forward(self,prob,label):
        prob = torch.log_softmax(prob, dim=1)
        # B C
        entropy = - (prob.exp() * prob).sum(dim=1).unsqueeze(1) # B
        entropy = self.threshold - entropy
        mask = entropy.gt(0)
        return (self.nll(prob,label) - self.beta*entropy*mask).mean()

"""
refer to: 
Maximum Entropy on Erroneous Predictions (MEEP):
Improving model calibration for medical image segmentation Agostina
"""

class AguEntropyLoss(nn.Module):
    def __init__(self,beta=0.1):
        super(AguEntropyLoss, self).__init__()
        self.beta = beta
        self.nll = nn.NLLLoss()

    def forward(self,prob,label):
        # B C
        prob = torch.log_softmax(prob, dim=1)
        mask = (torch.max(prob,dim=1)[1]!=label)
        entropy = -((prob.exp() * prob).sum(dim=1)[mask]).mean()

        return self.nll(prob,label) - self.beta*entropy,entropy


class KLEntropyLoss(nn.Module):
    def __init__(self,n_class = 100, beta=0.1):
        super(KLEntropyLoss, self).__init__()
        self.beta = beta
        self.n_class = n_class
        self.nll = nn.NLLLoss()

    def forward(self,prob,label):
        prob = torch.log_softmax(prob,dim=1)
        mask = (torch.max(prob, dim=1)[1] != label)
        """
        KL divergence
        """
        KL_D = ((1/self.n_class)* torch.log((1/self.n_class)/(prob.exp()[mask,:]+1e-10)).sum(dim=1)) # B
        # print(KL_D.mean())
        return self.nll(prob,label) - self.beta*KL_D.mean(), KL_D.mean()


def L2_Re(model, beta):
    L2 = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            L2 += module.weight.data.abs().sum()

    return L2*beta

if __name__=="__main__":
    re = AguEntropyLoss()
    por = torch.rand([128,100])
    la = torch.randint(0,100,[128])
    _,KL = re(por,la)
    print(KL)
    print(_)


