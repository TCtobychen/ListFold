import numpy as np 
import os
import math
import argparse
import pandas as pd 
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch_model import CMLE, LMLE, Closs, Closs_explained, Closs_sigmoid

Models = {'CMLE': CMLE, 'LMLE': LMLE}
Losses = {'plain': Closs, 'explained': Closs_explained, 'sigmoid': Closs_sigmoid}

parser = argparse.ArgumentParser(description='Rolling Training')
parser.add_argument(
    '--train-rolling-length',
    type=float,
    default=300,
    help='rolling length of training')
parser.add_argument(
    '--test-rolling-length',
    type=float,
    default=16,
    help='rolling length of test')
parser.add_argument(
    '--epochs',
    type=int,
    default=1000,
    help='train epochs')
parser.add_argument(
    '--model-type',
    type=str,
    default='CMLE',
    help='model type, either CMLE or LMLE')
parser.add_argument(
    '--loss-type',
    type=str,
    default='explained',
    help='loss type, either explained, sigmoid or LMLE_loss')
parser.add_argument(
    '--N',
    type=int,
    default=20,
    help='number of rolling pairs')
parser.add_argument(
    '--nfeatures',
    type=int,
    default=68,
    help='number of features')
parser.add_argument(
    '--batch-size',
    type=int,
    default=32,
    help='batch size used in training')
parser.add_argument(
    '--short',
    type=str,
    default='bottom',
    help='')

args = parser.parse_args()
suffix = str(args.train_rolling_length) + '_' + str(args.test_rolling_length)

batch_size = 32

def return_rank(a):
    a = a * -1
    order = a.argsort()
    return order.argsort()


def test(model, test_features):
    test_features = np.load(test_features, allow_pickle = True)
    print(test_features.shape)
    #print(len(test_features))
    L = len(test_features)
    N = len(test_features) // batch_size + 1
    v = np.zeros((N*batch_size, test_features.shape[1], test_features.shape[2]))
    v[:L, :, :] = test_features
    for i in range(N * batch_size - L):
        v[i+L,:,:] = test_features[0,:,:]
    res = []
    for i in range(N):
        batch_x = Variable(torch.from_numpy(v[i * batch_size:(i+1) * batch_size,:,:]).double())
        scores = model(batch_x)
        res.append(np.array(scores.data.cpu()))
    res = np.concatenate(res, axis = 0)
    res = res[:L]
    return res

def back_test(k, score, returns):
    res = []
    weight_list_pos, weight_list_neg = [], []
    return_list_pos, return_list_neg = [], []
    for i in range(len(score)):
        rank = return_rank(score[i])
        rank2ind = np.zeros(len(rank), dtype = int)
        for j in range(len(rank)):
            rank2ind[rank[j]] = j
        weights = np.zeros(k)
        for j in range(k):
            weights[j] = score[i][rank2ind[j]]
        s = k * (k+1) / 2.0
        for j in range(k):
            weights[j] = (k - j) / s
            weights[j] = 1.0 / k
        total_return = 0
        for j in range(k):
            total_return += weights[j] * returns[i][rank2ind[j]]
            total_return -= weights[j] * returns[i][rank2ind[79 - j]]
        for j in range(80):
            total_return += 0#1.0/80.0 * returns[i][rank2ind[j]]
        res.append(total_return)
        pos, neg = [], []
        r_pos, r_neg = [], []
        for j in range(k):
            pos.append(rank2ind[j])
            neg.append(rank2ind[79 - j])
            r_pos.append(returns[i][rank2ind[j]])
            r_neg.append(returns[i][rank2ind[79 - j]])
        weight_list_pos.append(pos)
        weight_list_neg.append(neg)
        return_list_pos.append(r_pos)
        return_list_neg.append(r_neg)
    return np.array(res), np.array(weight_list_pos), np.array(weight_list_neg), np.array(return_list_pos), np.array(return_list_neg)
    return np.array(res), 1

def back_test_all_rank(k, score, returns):
    rp = np.zeros((len(score), 80))
    rt = np.zeros((len(score), 80))
    for i in range(len(score)):
        rp[i] = return_rank(score[i])        
        rt[i] = return_rank(returns[i])
    return rp, rt 


        

def back_test_rank(k, score, returns):
    ranks = []
    for i in range(len(score)):
        rank = return_rank(score[i])
        rank2ind = np.zeros(len(rank), dtype = int)
        for j in range(len(rank)):
            rank2ind[rank[j]] = j
        ranks.append(rank2ind)
    return ranks

def total_return(a):
    ans = 1
    for item in a:
        ans *= (1 + item)
    return ans

def load_model_test(model, model_name, test_features, test_ranks):
    tmp = []
    saved_state = torch.load(model_name)
    model.load_state_dict(saved_state)
    y_pred = test(model, test_features)
    y = np.array(y_pred)
    y_pred = test(model, test_features)
    y = np.array(y_pred)
    r = np.load(test_ranks)
    r = r[:len(y), :]
    
    for j in range(80):
        res, _ = back_test(j+1, y, r)
        tmp.append(np.mean(res))
    
    #res, _ = back_test(8, y ,r)
    return tmp

def load_model_test_rank(model, model_name, test_features, test_ranks):
    tmp = []
    saved_state = torch.load(model_name)
    model.load_state_dict(saved_state)
    y_pred = test(model, test_features)
    y = np.array(y_pred)
    y_pred = test(model, test_features)
    y = np.array(y_pred)
    r = np.load(test_ranks)
    r = r[:len(y), :]
    res, weight_list_pos, weight_list_neg, return_list_pos, return_list_neg = back_test(8, y, r)
    return res, weight_list_pos, weight_list_neg, return_list_pos, return_list_neg

def load_model_test_all_rank(model, model_name, test_features, test_ranks, ranks):
    tmp = []
    saved_state = torch.load(model_name)
    model.load_state_dict(saved_state)
    y_pred = test(model, test_features)
    y = np.array(y_pred)
    y_pred = test(model, test_features)
    y = np.array(y_pred)
    r = np.load(test_ranks)
    r = r[:len(y), :]
    rp, rt = back_test_all_rank(8, y, r)
    return rp, rt


Model = Models[args.model_type]
model_t = Model(n_features = args.nfeatures)
model_t = model_t.double()
'''
t = []
for ind in range(0, 7):
    d = pd.DataFrame()
    for itr in range(1999, 2000):
        print(ind, itr)
        tmp = load_model_test(model_t, 'rolling_model_' + str(ind) + '_' + str(itr) + '.json', './rolling/features_test_' + str(ind) + '.npy', './rolling/ranks_test_' + str(ind) + '.npy')
        t.append(tmp)
t = np.concatenate(t, axis = 0)
d = pd.DataFrame()
d['return'] = t
d.to_csv('rolling_returns.csv', index = True) 
'''

def load_model_test_ranks(model, model_name, test_features, test_ranks, ranks):
    tmp = []
    saved_state = torch.load(model_name)
    model.load_state_dict(saved_state)
    y_pred = test(model, test_features)
    y = np.array(y_pred)
    y_pred = test(model, test_features)
    y = np.array(y_pred)
    r = np.load(test_ranks)
    r = r[:len(y), :]
    res, weight_list_pos, weight_list_neg, return_list_pos, return_list_neg = back_test(ranks,y,r)
    
    return res, weight_list_pos, weight_list_neg, return_list_pos, return_list_neg
'''
epoch_to_use = 999
rank_to_use = 8

RP, RT = [], []
for ind in range(0, args.N):
    for itr in range(epoch_to_use, epoch_to_use+1):
        print(ind, itr)
        rp, rt = load_model_test_all_rank(model_t, './models_' + suffix + '_' + args.loss_type + '/rolling_model_' + str(ind) + '_rvs' + str(itr) + '.dat', './rolling_' + suffix + '/features_test_' + str(ind) + '.npy', './rolling_' + suffix +'/ranks_test_' + str(ind) + '.npy', rank_to_use)
        RP.append(rp)
        RT.append(rt)
RP = np.concatenate(RP, axis=0)
RT = np.concatenate(RT, axis=1)
print('here')
np.save('./results_' + suffix + '/results_' + args.loss_type + '_' + str(epoch_to_use) + '_' + str(rank_to_use)+'_rvs_pred.npy', RP)
np.save('./results_' + suffix + '/results_' + args.loss_type + '_' + str(epoch_to_use) + '_' + str(rank_to_use)+'_rvs_true.npy', RT)
'''



d = pd.DataFrame()
tt = []
wp, wn, rp, rn = [], [], [], []
epoch_to_use = 999
rank_to_use = 8

for ind in range(0, args.N):
    for itr in range(epoch_to_use, epoch_to_use+1):
        print(ind, itr)
        tmp, weight_list_pos, weight_list_neg, return_list_pos, return_list_neg = load_model_test_ranks(model_t, './models_' + suffix + '_' + args.loss_type +'/rolling_model_' + str(ind) + f'_{args.batch_size}_' + str(itr) + '.dat', './rolling_' + suffix + '/features_test_' + str(ind) + '.npy', './rolling_' + suffix +'/ranks_test_' + str(ind) + '.npy', rank_to_use)
        tt.append(tmp)
        wp.append(weight_list_pos)
        wn.append(weight_list_neg)
        rp.append(return_list_pos)
        rn.append(return_list_neg)

if not os.path.exists('./results_' + suffix):
    os.makedirs('./results_' + suffix)
tt = np.concatenate(tt)
wp = np.concatenate(wp, axis=0)
wn = np.concatenate(wn, axis=0)
rp = np.concatenate(rp, axis=0)
rn = np.concatenate(rn, axis=0)
for i in range(8):
    d['pos_ticker_'+str(i+1)] = wp[:,i]
for i in range(8):
    d['neg_ticker_'+str(i+1)] = wn[:,i]
d['return'] = tt
d.to_csv('./results_' + suffix +'/results_' + args.loss_type + '_' + str(epoch_to_use) + '_' + str(rank_to_use)+f'_{args.batch_size}.csv', index = False)

