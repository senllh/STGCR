import argparse
import pickle
import time
from util import Data, split_validation, Get_sessGraph
from model import *
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='retailrocket', help='dataset name: diginetica/retailrocket/sample')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--embSize', type=int, default=100, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--layer', type=float, default=2, help='the number of layer used')  #3
    parser.add_argument('--beta', type=float, default=1e-5, help='ssl task maginitude')  # 0.01 [0.1, 1, 3, 5, 10, 20, 50] 1e-5
    parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')
    parser.add_argument('--n_time', type=int, default=5, help='the number of time slices')
    parser.add_argument('--decay', type=float, default=0.7, help='time decay')
    arg = parser.parse_args()
    return arg

opt = parse_args()

def main():
    train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('./datasets/' + opt.dataset + '/short.txt', 'rb'))

    if opt.dataset == 'diginetica':
        n_node = 43097
    elif opt.dataset == 'retailrocket':
        n_node = 36968
    elif opt.dataset == 'yoochoose1_64':
        n_node = 37483
    else:
        n_node = 309

    train_data = Data(train_data, shuffle=True, n_node=n_node)
    test_data = Data(test_data, shuffle=False, n_node=n_node)

    model = trans_to_cuda(STGCR(adjacency=train_data.adjacency, n_node=n_node, lr=opt.lr, l2=opt.l2, beta=opt.beta, layers=opt.layer, emb_size=opt.embSize, batch_size=opt.batchSize,dataset=opt.dataset))

    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))

if __name__ == '__main__':
    main()
