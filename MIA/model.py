import torch
import torch.nn as nn


"""
This model is referred to paper:

"Machine Learning with Membership Privacy using Adversarial Regularization"

More detail can be found in:
https://dl.acm.org/doi/abs/10.1145/3243734.3243855

this code is implemented in 2022.01.03 

version : v1

"""

class Adversary(nn.Module):
    def __init__(self,n_class=100):
        super(Adversary, self).__init__()
        self.n_class = n_class

        # for prediction
        self.pred_fc = nn.Sequential(nn.Linear(self.n_class,1024),
                                     nn.ReLU(),
                                     nn.Linear(1024,512),
                                     nn.ReLU(),
                                     nn.Linear(512,64),
                                     nn.ReLU())
        # for label
        self.label_fc = nn.Sequential(nn.Linear(self.n_class,512),
                                      nn.ReLU(),
                                      nn.Linear(512,64),
                                      nn.ReLU())

        # fuse layer
        self.class_layer = nn.Sequential(nn.Linear(128,256),
                                        nn.ReLU(),
                                        nn.Linear(256,64),
                                        nn.ReLU(),
                                        nn.Linear(64,1))

        # init weight
        for key in self.state_dict():
            if key.split('.')[-1] == 'weight':
                nn.init.normal_(self.state_dict()[key], std=0.01)

            elif key.split('.')[-1] == 'bias':
                self.state_dict()[key][...] = 0

    def forward(self,x,y):
        # x should be softmax output

        x1 = self.pred_fc(x) # B C
        x2 = self.label_fc(y)
        x12 = torch.cat([x1,x2],dim=1)
        # x12 = self.bn(x12)
        out = self.class_layer(x12)
        out = torch.sigmoid(out)
        return out

    def init_weights(self,m):
        if isinstance(m,nn.Linear):
            m.weight.data.normal_(0,0.01)
            if m.bias.data is not None:
                m.bias.data.fill_(0)


# if __name__=="__main__":
#     ad = Adversary().train()
#     x = torch.randn([256,100])
#     index = torch.randint(0,100,[256]).unsqueeze(dim=1)
#     y = torch.zeros([256,100]).scatter_(value=1,index=index,dim=1)
#     out = ad(x,y)
#     print(out)
