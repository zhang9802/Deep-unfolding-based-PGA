"""
deep unfloed projection gradient descent algorithm to deal with the optimization problem with constant modulus constraint
"""
import argparse
from collections import namedtuple
from itertools import count
import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
# from tensorboardX import SummaryWriter
import copy
import matplotlib.pyplot as plt
import scipy.io as sio

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
class ComplexReLU(nn.Module):
    def __init__(self):
        super(ComplexReLU, self).__init__()

    def forward(self, input):
        return torch.relu(input.real)+ 1j*torch.relu(input.imag)  # 对实部和虚部分别应用ReLU

def whiten(state):
    return (state - np.mean(state)) / np.std(state)

def isnan(x):
    return x != x

class UnfoldedModel(nn.Module):
    """
    Deep unfolding for PGA algorithm
    """
    def __init__(self,J):
        """
        :param alpha:  learnable step size, shape: (J,1)
        :param J: number of layers of the unfolded PAG network
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.normal(0.01, 0.1 * torch.ones((J,1))))
        self.J = J
        # self.complexRelu = ComplexReLU()

    def forward(self, H, X):
        """
        :param H: channels, shape: batch_size * Nt, dtype=torch.complex
        :param X: initial points, shape: batch_size * Nt, dtype=torch.complex
        :return:
        """
        # H = H.detach()
        # X = X.detach()
        Rs = torch.zeros(self.J, dtype=torch.float64).to(device)
        # iterations
        for jj in range(self.J):
            Grad_X = grad_x(H, X).to(device)  # calculate gradient
            # print(Grad_X)
            X = X - self.alpha[jj] * Grad_X   # gradient descent
            # X = self.complexRelu(X)
            # X = torch.exp(1j * torch.angle(X))  # projecting into the constant modulus constraint
            X = X / torch.norm(X, p=2, dim=(1,), keepdim=True).repeat(1,X.shape[1])
            # temp = torch.sqrt(X[:,0:int(X.shape[1]/2)] **2 + X[:,int(X.shape[1]/2):X.shape[1]] **2).repeat((1,2))
            # X = X / temp
            Rs[jj] = torch.sum(H * X)  

        return X, Rs


def grad_x(H,X):

    """
    :param H: channels, shape: batch_size * Nt, dtype=torch.complex
    :param X: initial points, shape: batch_size * Nt, dtype=torch.complex
    :return: grad with respect to X
    """
    Grad_X = torch.zeros(X.shape, dtype=torch.float64)
    # iterations on each sample
    for k in range(H.shape[0]):
        Grad_X[k,:] =  2 * H[k,:]

    return Grad_X

def train_nework(H, X, device, Nt, learning_rate, num_eps, batch_size, J):
    # Start training:
    print("Unfolded starts training")
    n_samples, Nt = H.shape
    steps_per_epoch = n_samples // batch_size
    # alpha = torch.full((J, 1), 0.001, dtype=torch.float64, device=device, requires_grad=True)
    # build the optimizer and criterion

    # criterion = nn.MSELoss()
    unfolded_model = UnfoldedModel(J).to(device)
    # optimizer = torch.optim.SGD([alpha], lr=learning_rate,  momentum=0.9)
    optimizer = torch.optim.Adam(unfolded_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_list = []
    for epoch in range(num_eps):
        index_samples = np.random.choice(a=n_samples, size=n_samples, replace=False, p=None)
        H_shuffle = H[index_samples,:]
        X_shuffle = X[index_samples,:]

        for step in range(steps_per_epoch):
            H_batch = H_shuffle[step * batch_size:(step + 1) * batch_size, :]
            X_batch = X_shuffle[step * batch_size:(step + 1) * batch_size, :]

            # get the outputs
            X_h, Rs = unfolded_model.forward(H_batch, X_batch)
            # compute the losss
            loss = torch.sum(Rs)
            # loss = torch.sum(torch.abs(H_batch.conj() * X_h)**2)

            # paramapter optimizations
            optimizer.zero_grad()
            loss.backward()
            grads = [param.grad for param in unfolded_model.parameters()]

            grads_alpha = grads[0]
            # print(grads_alpha.data)
            # print(grads_alpha.grad)
            if isnan(grads_alpha).any():  # avoiding NaN in gradients
                print("NaN_grad")
                continue
            optimizer.step()

            # print(alpha)
            # print(alpha.grad)
            with torch.no_grad():
                loss_list.append(torch.sum(Rs).data.cpu()/batch_size)
            # print(Rs[-1])
        print(f"epoch = {epoch}, loss = {loss_list[-1].detach().data:.4f}")
    return loss_list, unfolded_model


if __name__ == '__main__':
    # definite some parametes
    parser = argparse.ArgumentParser()
    parser.add_argument("--Nt", default=5, type=int, metavar='N', help='Number of antennas in the BS')
    parser.add_argument("--max_iterations", default=5, type=int, metavar='N', help='Number of iterations')
    parser.add_argument("--num_eps", default=1000, type=int, metavar='N',help='Maximum number of episodes (default: 5000)')
    parser.add_argument("--batch_size", default=32, type=int,metavar='N', help='Batch size (default: 16)')
    parser.add_argument("--data_size", default=100, type=int,metavar='N', help='Batch size (default: 16)')
    parser.add_argument("--learning_rate", default="1e-3", type=float, help='dims of hidden layer')
    parser.add_argument("--gpu", default="0", type=int, help='gpu or cpu')

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(device)
    np.random.seed(0)  # numpy seed random seed
    torch.manual_seed(0)  # torch seed random seed
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(0)  # cuda seed random seed

    # generate coefficient matrix
    H = np.random.rand(args.data_size, args.Nt)   # coefficient matrix
    # H = whiten(H)
    H = torch.from_numpy(H).to(torch.float64).to(device)
    #generate initial points
    # X = torch.full((args.data_size, args.Nt), 1, dtype=torch.complex128, device=device)
    X = np.random.randn(args.data_size, args.Nt)    # coefficient matrix
    # X = whiten(X)
    X = torch.from_numpy(X).to(torch.float64).to(device)
    # optimization variable : alpha, learnable stepsize
    # alpha = torch.full((args.max_iterations, 1), 0.1, dtype=torch.float64, device=device, requires_grad=True)
    # save loss
    err_list, unfolded_model = train_nework(H, X, device, args.Nt, args.learning_rate, args.num_eps, args.batch_size, args.max_iterations)

    #plot loss curve
    plt.figure()
    plt.plot(err_list,'r-',linewidth = 1)
    plt.grid()
    plt.title('learning rate = '+ str(args.learning_rate) + ' batch size = ' + str(args.batch_size))
    plt.xlabel('episode')
    plt.ylabel('loss')

    # plt.show()
    name = 'loss_' + str(args.learning_rate) + '_' + str(args.batch_size) + 'V2' + '.png'
    plt.savefig(name)

    # test phase

    # H_test = np.random.rand(1, args.Nt)  # coefficient matrix
    # sio.savemat('file.mat',{"H":H_test})
    # H_test = torch.from_numpy(H_test).to(torch.float64)
    # #generate initial points
    # X_test = np.random.randn(1, args.Nt)    # coefficient matrix
    # X_test = torch.from_numpy(X_test).to(torch.float64)
    # with torch.no_grad():
    #     unfolded_model = UnfoldedModel(alpha, args.max_iterations).to(device)
    #     X_output, Rs_output = unfolded_model.forward(H_test, X_test)
    test_num = 100

    H_test = np.random.rand(test_num, args.Nt)  # coefficient matrix
    # H_test = whiten(H_test)
    X_test = np.random.randn(test_num, args.Nt)  # coefficient matrix
    # X_test = whiten(X_test)
    H_test = torch.from_numpy(H_test).to(torch.float64).to(device)
    # generate initial points
    X_test = torch.from_numpy(X_test).to(torch.float64).to(device)
    with torch.no_grad():
        for ii in range(test_num):

            # sio.savemat('file.mat',{"H":H_test})

            X_output, Rs_output = unfolded_model.forward(H_test[ii].reshape(1,-1), X_test[ii].reshape(1,-1))
            print(torch.sum(X_output * H_test[ii].reshape(1,-1)).data + torch.norm(H_test[ii].reshape(1,-1),p=2).data)  # torch.linalg.norm(H_test).data: optimal value, X_output*H_test).data optimal value correspinding to the output of the network
