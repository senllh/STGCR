import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
import torch_geometric.nn as gnn
import torch_geometric.utils as g_utils
from scipy import sparse
from tqdm import tqdm
import pickle
from main import parse_args

opt = parse_args()

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def to_hyper(data):
    hyper_e = len(data)
    item_id = []
    sess_id = []
    for sess in range(hyper_e):
        for item in data[sess]:
            item_id.append(item-1)
            sess_id.append(sess)
    return sparse.coo_matrix((np.ones(len(item_id), dtype=int), (item_id, sess_id)))#torch.stack([item_id, sess_id], dim=0)  # sparse.coo_matrix(hyper_graph)

class Hyperconv(nn.Module):
    def __init__(self, emb_size, activation, num_layers):
        super(Hyperconv, self).__init__()
        self.activation = activation()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(gnn.HypergraphConv(emb_size, emb_size, heads=1, concat=False, negative_slope=0.2, dropout=0.5,
                                         bias=True)).cuda()
            self.batch_norms.append(nn.BatchNorm1d(emb_size))

    def forward(self, x, edge_index):
        H_item = []
        for conv, bn in zip(self.layers, self.batch_norms):
            h_item = conv(x, edge_index)
            # h_item = self.activation(h_item)
            # h_item = bn(h_item)
            h_item = F.normalize(h_item, dim=-1, p=2)
            H_item.append(h_item)
        H_item = torch.stack(H_item, dim=1).sum(dim=1)
        return H_item

class GCN(nn.Module):
    def __init__(self, emb_size, activation, num_layers):
        super(GCN, self).__init__()
        self.activation = activation()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(gnn.GCNConv(emb_size, emb_size)).cuda()
            # self.layers.append(gnn.GraphSAGE(emb_size, emb_size, 1)).cuda()
            self.batch_norms.append(nn.BatchNorm1d(emb_size))

    def forward(self, x, edge_index, edge_weight, weight):
        H_item = []
        for conv, bn in zip(self.layers, self.batch_norms):
            if weight == True:
                h_item = conv(x=x, edge_index=edge_index, edge_weight=edge_weight).to(torch.float32)
            else:
                h_item = conv(x=x, edge_index=edge_index)
            # h_item = self.activation(h_item)
            # h_item = bn(h_item)
            h_item = F.normalize(h_item, dim=-1, p=2)
            H_item.append(h_item)
        H_item = torch.stack(H_item, dim=1).sum(dim=1)
        return H_item

class Cross_view_STG(Module):
    def __init__(self, layers, dataset, emb_size=100):
        super(Cross_view_STG, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset
        self.HyEdge_index = []
        self.GloEdge_index = []
        self.GloEdge_data = []
        n_time = opt.n_time

        # STHG Data
        for time in range(n_time, 0, -1):
            d_session = pickle.load(open('datasets/' + dataset + '/time_slice' + str(time) + '.txt', 'rb'))
            # d_session = pickle.load(open('datasets/' + dataset + '/train.txt', 'rb'))
            hyper_d = to_hyper(d_session)# [0]
            hyper_d = g_utils.from_scipy_sparse_matrix(hyper_d)
            self.HyEdge_index.append(hyper_d[0].to('cuda'))
        self.HYGonv = nn.ModuleList()

        for i in range(n_time):
            self.HYGonv.append(Hyperconv(emb_size=self.emb_size, activation=nn.LeakyReLU, num_layers=opt.layer))

        # STG Data
        for time in range(n_time, 0, -1):
            Global_w = sparse.load_npz('datasets/' + dataset + '/time_slice' + str(time) + '.npz')
            # Global_w = sparse.load_npz('datasets/' + dataset + '/Global.npz')
            Global_w = Global_w.multiply(1.0 / Global_w.sum(axis=0).reshape(1, -1))
            Global_w = g_utils.from_scipy_sparse_matrix(Global_w)
            self.GloEdge_index.append(Global_w[0].to('cuda'))
            self.GloEdge_data.append(Global_w[1].to('cuda'))
        self.GCNConv = nn.ModuleList()

        for i in range(n_time):
            self.GCNConv.append(GCN(emb_size=self.emb_size, activation=nn.LeakyReLU, num_layers=opt.layer))

        self.GRU1 = torch.nn.GRU(input_size=emb_size, hidden_size=emb_size, batch_first=True, bidirectional=False).cuda()
        self.GRU2 = torch.nn.GRU(input_size=emb_size, hidden_size=emb_size, batch_first=True, bidirectional=False).cuda()

    def forward(self, embedding):
        # STG
        G_w = []
        for conv, edge_index, edge_weight in zip(self.GCNConv, self.GloEdge_index, self.GloEdge_data):
            G_w.append(conv(embedding, edge_index, edge_weight, weight=True))
        G_w = torch.stack(G_w, dim=1)
        h1, x1 = self.GRU1(G_w)
        x1 = x1.squeeze(0)

        # STH
        H_w = []
        for Hyconv, edge_index in zip(self.HYGonv, self.HyEdge_index):
            H_w.append(Hyconv(embedding, edge_index))
        H_w = torch.stack(H_w, dim=1)
        h2, x2 = self.GRU2(H_w)
        x2 = x2.squeeze(0)
        return x1, x2, h1, h2

class STG_Co(nn.Module):
    def __init__(self, emb_size, dataset):
        super(STG_Co, self).__init__()
        self.emb_size = emb_size
        self.fc1 = nn.Sequential(nn.Linear(emb_size, emb_size), nn.LeakyReLU(),  nn.Linear(emb_size, emb_size))
        self.fc2 = nn.Sequential(nn.Linear(emb_size, emb_size), nn.LeakyReLU(),  nn.Linear(emb_size, emb_size))
        self.n_time = opt.n_time
        self.decay = opt.decay

    def forward(self, z1, z2):

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding

        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        def co_loss(h1, h2):
            pos = score(h1, h2)  # 正样本是 两个视角的交互
            neg1 = score(h1, row_column_shuffle(h2))  # 负样本是 打乱交互
            one = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1)
            sub_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg1))))
            return sub_loss

        con_loss = 0
        for w in range(self.n_time):
            #  node - node
            h1 = self.fc1(z1.permute([1, 0, 2])[w])  # project layer
            h2 = self.fc2(z2.permute([1, 0, 2])[w])
            sub_loss = co_loss(h1, h2)
            con_loss += sub_loss * self.decay**(w-1)
        return con_loss * opt.beta

class STGCR(Module):
    def __init__(self, adjacency, n_node, lr, layers, l2, beta, dataset, emb_size=100, batch_size=100):
        super(STGCR, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta
        self.adjacency = adjacency
        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.pos_embedding = nn.Embedding(300, self.emb_size)
        self.HyperGraph = Cross_view_STG(self.layers, dataset)
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()
        self.STG_Co = STG_Co(self.emb_size, dataset)

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def generate_sess_emb(self, item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, seq_h], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        return select

    def forward(self, session_item, session_len, reversed_sess_item, mask):
        g, h, G, H = self.HyperGraph(self.embedding.weight)
        co_loss = self.STG_Co(H, G)
        g = g + h
        S = self.generate_sess_emb(g, session_item, session_len, reversed_sess_item, mask)
        return g, S, co_loss

def forward(model, i, data):
    tar, session_len, session_item, reversed_sess_item, mask = data.get_slice(i)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    item_emb_hg, sess_emb_hgnn, con_loss = model(session_item, session_len, reversed_sess_item, mask)
    scores = torch.mm(sess_emb_hgnn, torch.transpose(item_emb_hg, 1, 0))
    return tar, scores, con_loss

def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    model.train()
    slices = train_data.generate_batch(model.batch_size)
    for i in tqdm(slices):
        model.zero_grad()
        targets, scores, con_loss = forward(model, i, train_data)
        loss = model.loss_function(scores + 1e-8, targets)
        loss = loss + con_loss
        loss.backward()
#        print(loss.item())
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        tar, scores, con_loss = forward(model, i, test_data)
        scores = trans_to_cpu(scores).detach().numpy()
        index = np.argsort(-scores, 1)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' %K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' %K].append(0)
                else:
                    metrics['mrr%d' %K].append(1 / (np.where(prediction == target)[0][0]+1))
    return metrics, total_loss



