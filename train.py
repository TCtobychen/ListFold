import numpy as np 
import os
import argparse
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch_model import CMLE, LMLE, Closs, Closs_explained, Closs_sigmoid, Lloss, learning_rate

Models = {'CMLE': CMLE, 'LMLE': LMLE}
Losses = {'plain': Closs, 'explained': Closs_explained, 'sigmoid': Closs_sigmoid, 'LMLE_loss': Lloss}


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
    '--pp',
    type=int,
    nargs='+',
    default=21,
    help='parameters for parallel training')
parser.add_argument(
    '--nfeatures',
    type=int,
    default=68,
    help='number of features')
parser.add_argument(
    '--batch-size',
    type=int,
    default=32)
args = parser.parse_args()
batch_size = args.batch_size

def return_rank(a):
    a = a * -1
    order = a.argsort()
    return order.argsort()

def random_batch(x, y):
	ind = np.random.randint(0, len(x), batch_size)
	batch_x, batch_y = x[ind], y[ind]
	x_sorted = np.zeros(batch_x.shape)
	for i in range(len(batch_x)):
		rank_temp = return_rank(batch_y[i])
		rank2ind = np.zeros(80, dtype = int)
		for j in range(len(rank_temp)):
			rank2ind[rank_temp[j]] = int(j)
		for j in range(len(rank_temp)):
			x_sorted[i,rank_temp[j],:] = batch_x[i][rank2ind[rank_temp[j]]]
	return x_sorted


def train(features, ranks, epochs, model_name, args):
	features = np.load(features, allow_pickle = True)
	ranks = np.load(ranks, allow_pickle = True)
	print('Done reading data\n')
	Model = Models[args.model_type]
	model = Model(n_features = args.nfeatures)
	model = model.double()
	loss = Losses[args.loss_type]()
	opt = optim.Adam(model.parameters(), lr=learning_rate[args.loss_type])
	print('Done building model\n')
	running_loss = []
	torch.set_grad_enabled(True)
	for itr in range(epochs):
		batch_x = Variable(torch.from_numpy(random_batch(features, ranks)).double())
		model.train()
		scores = model(batch_x)
		l = loss(scores, torch.tensor(80, requires_grad = False))
		opt.zero_grad()
		l.backward()
		opt.step()
		running_loss.append(float(l))
		if (itr+1) % epochs == 0:
			print("step", (itr+1), np.mean(running_loss))
			running_loss = []
			torch.save(model.state_dict(), model_name + str(itr) + '.dat')

P = args.pp
suffix = str(args.train_rolling_length) + '_' + str(args.test_rolling_length)
if not os.path.exists('./models_' + suffix + '_' + args.loss_type):
	os.makedirs('./models_' + suffix + '_' + args.loss_type)
if len(P) == 1:
	N = P[0]
	for ind in range(0, N):
		print(ind)
		train('./rolling_' + suffix + '/features_train_' + str(ind) + '.npy', './rolling_' + suffix + '/ranks_train_' + str(ind) + '.npy', args.epochs, './models_' + suffix + '_' + args.loss_type + '/rolling_model_' + str(ind) + '_'+str(batch_size) + '_', args) 
if len(P) == 2:
	N, m = P[0], P[1]
	for ind in range(0, N):
		if ind % 3 != m:
			continue
		print(ind)
		train('./rolling_' + suffix + '/features_train_' + str(ind) + '.npy', './rolling_' + suffix + '/ranks_train_' + str(ind) + '.npy', args.epochs, './models_' + suffix + '_' + args.loss_type + '/rolling_model_' + str(ind) + '_'+str(batch_size) + '_', args) 

