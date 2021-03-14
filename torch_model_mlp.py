import torch
from torch import nn, optim
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        m.weight.data.normal_(mean = 0.0, std = 0.05)
        m.bias.data.fill_(0.05)

class Closs(nn.Module):
    def __init__(self):
        super(Closs, self).__init__()
    def forward(self, f, num_stocks):
        l = torch.sum(f[:,num_stocks // 2:], dim = 1) - torch.sum(f[:, :num_stocks // 2], dim = 1)
        for i in range(num_stocks // 2):
            l += torch.logsumexp(f[:,i:num_stocks-i], dim = 1)
            l += torch.logsumexp(torch.neg(f[:,i:num_stocks-i]), dim = 1)
        l = torch.mean(l)
        return l

class Closs_explained(nn.Module):
    def __init__(self):
        super(Closs_explained, self).__init__()
    def forward(self, f, num_stocks):
        l = torch.sum(f[:,num_stocks // 2:], dim = 1) - torch.sum(f[:, :num_stocks // 2], dim = 1)
        for i in range(num_stocks // 2):
            subtract = torch.tensor(num_stocks - 2*i,requires_grad = False)
            l += torch.log(torch.sum(torch.exp(f[:,i:num_stocks-i]), dim = 1)*torch.sum(torch.exp(torch.neg(f[:,i:num_stocks-i])), dim = 1)-subtract)
        l = torch.mean(l)
        return l

class Closs_sigmoid(nn.Module):
    def __init__(self):
        super(Closs_sigmoid, self).__init__()
    def forward(self, f, num_stocks):
        l = torch.tensor(1, requires_grad=False)+torch.exp(f[:,num_stocks//2:] - f[:,:num_stocks//2])
        return torch.mean(torch.log(l))

class Lloss(nn.Module):
    def __init__(self):
        super(Lloss, self).__init__()
    def forward(self, f, num_stocks):
        l = torch.neg(torch.sum(f, dim = 1))
        for i in range(num_stocks):
            l += torch.logsumexp(f[:,i:], dim = 1)
        l = torch.mean(l)
        return l

class CMLE(nn.Module):
    def __init__(self, n_features):
        super(CMLE, self).__init__()
        self.n_features = n_features
        self.linear1 = nn.Linear(self.n_features, self.n_features * 4)
        self.linear2 = nn.Linear(self.n_features * 4, self.n_features * 2)
        self.linear3 = nn.Linear(self.n_features * 2, self.n_features // 2)
        self.linear4 = nn.Linear(self.n_features // 2, 1)
        self.apply(weights_init)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        result = self.linear4(x)
        result = result.view(result.shape[0], result.shape[1])
        return result

class LMLE(nn.Module):
    def __init__(self, n_features, num_stocks):
        super(CMLE, self).__init__()
        self.n_features = n_features
        self.num_stocks = num_stocks
        self.linear1 = nn.Linear(self.n_features, self.n_features * 4)
        self.linear2 = nn.Linear(self.n_features * 4, self.n_features * 2)
        self.linear3 = nn.Linear(self.n_features * 2, sefl.n_features // 2)
        self.linear4 = nn.Linear(self.n_features // 2, 1)
        self.apply(weights_init)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        result = result.view(result.shape[0], result.shape[1])
        return result

learning_rate = {'explained': 5e-5, 'sigmoid': 1e-4, 'LMLE_loss': 1e-4}
