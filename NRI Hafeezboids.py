#!/usr/bin/env python
# coding: utf-8

# ## **Imports**
# 

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Dropout
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn import Sequential, Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch.utils.data as data_utils
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader


# In[2]:


import sys
print(sys.version)


# In[3]:


def load_data(batch_size=1):
    features_data = np.load('features_boids' + '.npy')
    edges_data = np.load('edges_data' + '.npy')
#     print(np.shape(features_data))
    features_data = features_data[:,:,0:4,:]
#     print(np.shape(features_data))
#     print(np.shape(edges_data))
    edges_data = edges_data[:100,:,:]
    
    # [num_samples, num_timesteps, num_dims, num_atoms]
    print(np.shape(edges_data))

    loc_train= features_data[:int(len(features_data)*0.8),:,0:2,:]
    loc_valid =features_data[int(len(features_data)*0.8):int(len(features_data)*0.9),:,0:2,:]
    loc_test = features_data[int(len(features_data)*0.9):,:,0:2,:]

    vel_train = features_data[:int(len(features_data)*0.8),:,2:4,:]
    vel_valid = features_data[int(len(features_data)*0.8):int(len(features_data)*0.9),:,2:4,:]
    vel_test =  features_data[int(len(features_data)*0.9):,:,2:4,:]

    edges_train= edges_data[:int(len(edges_data)*0.8),:,:]
    edges_valid =edges_data[int(len(edges_data)*0.8):int(len(edges_data)*0.9),:,:]
    edges_test = edges_data[int(len(edges_data)*0.9):,:,:]
#     print(np.shape(edges_train))

    num_atoms = 10
    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)



    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
#     print(np.shape(edges_train))
  
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)
    # print(np.shape(edges_train))
    # edges_train = edges_data

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    
    # next_feat_valid = feat_valid[:,:,1:,:]
    # feat_valid = feat_valid[:,:,0:-1,:]
    
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    
    feat_test = np.concatenate([loc_test, vel_test], axis=3)

    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)

    # print(edges_train.size())

    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]
    print(edges_train.size())
    print(feat_train.size(),'feature train')
        # [num_samples, num_atoms, num_timesteps, num_dims,]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size,shuffle= False)
    valid_data_loader = DataLoader(valid_data,  batch_size=batch_size,shuffle= False)
    test_data_loader = DataLoader(test_data,  batch_size=batch_size,shuffle= False)

    return train_data_loader, valid_data_loader, test_data_loader, loc_max, loc_min, vel_max, vel_min
train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(32)
# (x_sample,rel_sample) = next(iter(train_loader))
# print(x_sample.shape)
# print(rel_sample.shape)


# In[4]:


for data, relations in test_loader:
    try:
        g_truth = torch.cat((g_truth, data), dim=0)
    except:
        g_truth = data


# In[5]:


print(g_truth.size())
g_truth =g_truth.squeeze().detach().cpu().numpy().transpose(1,0,2,3).reshape(10,-1,4)
print(np.shape(g_truth))


# In[6]:


def normalize(data, data_max, data_min):
	return (data - data_min) * 2 / (data_max - data_min) - 1

# loc_max, loc_min, vel_max, vel_min
def urnnormalize(data, data_max, data_min):
	return (data + 1) * (data_max - data_min) / 2. + data_min



gt_position_unnormalize = urnnormalize(g_truth[:,:,0:2], loc_max, loc_min)
gt_velocity_unnormalize = urnnormalize(g_truth[:,:,2:4], vel_max, vel_min)

target_features =np.concatenate([gt_position_unnormalize, gt_velocity_unnormalize ],axis=2)
print(target_features.shape)

plt.figure()
plt.plot(target_features[0,:,0],target_features[0,:,1],  marker='o')


# ## **Utils**

# In[7]:


def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Taken from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input,dim=1)
    return soft_max_1d.transpose(axis, 0)


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):

    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y
def edge_accuracy(preds, target):
    _, preds = preds.max(-1)
    correct = preds.float().data.eq(
        target.float().data.view_as(preds)).cpu().sum()
    return float(correct) / (target.size(0) * target.size(1))


def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False,
                           eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))


# In[8]:


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob
        self.n_in = n_in
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


# ## **Encoder definition**

# In[9]:


class MLPEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(MLPEncoder, self).__init__()

        self.factor = factor

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t().float(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec.float(), x)
        senders = torch.matmul(rel_send.float(), x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        x = self.mlp1(x)  # 2-layer ELU net per node
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)

        return self.fc_out(x)


# ## **Decoder definition**

# In[10]:


class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False):
        super(MLPDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        pre_msg.size(2), self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps):
        # NOTE: Assumes that we have the same graph across all samples.

        inputs = inputs.transpose(1, 2).contiguous()

        sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
                 rel_type.size(2)]
        rel_type = rel_type.unsqueeze(1).expand(sizes)

        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]
        curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                 curr_rel_type)
            preds.append(last_pred)

        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                 preds[0].size(2), preds[0].size(3)]

        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()


# ## Training

# In[11]:


# Define GNN model
class GraphModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphModel, self).__init__()
        self.encoder = MLPEncoder(in_channels, hidden_channels, out_channels)
        self.decoder = MLPDecoder(n_in_node=4,
                         edge_types=2,
                         msg_hid=16,
                         msg_out=16,
                         n_hid=16, )
        
    def forward(self, x, rel_rec, rel_send,pred_steps):
        logits = self.encoder(x, rel_rec, rel_send)
        edges = gumbel_softmax(logits, tau=0.5, hard=False)
        output = self.decoder(x, edges, rel_rec, rel_send, pred_steps)
        return output, logits, edges
model = GraphModel(400, 256, 2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=model.to(device)


# In[12]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
# criterion = nn.MSELoss()
epochs = 2
num_atoms=10
import time

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot
off_diag = np.ones([num_atoms, num_atoms]) - np.eye(num_atoms)

rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)
rel_rec = rel_rec.to(device)
rel_send = rel_send.to(device)


# In[13]:


def LossNorm(y):
    y_1step= y[:,:,1:,:]
    y_prev = y[:,:,:-1,:]
    loss_sum = torch.abs(y_prev-y_1step)
#     print(torch.sum(loss_sum).size())    
    return torch.mean(loss_sum)
    
def MAELoss(y, y_hat, l_norm):
    mae = torch.abs(y - y_hat) 
    mae = torch.mean(mae)
    return mae/l_norm
def MAPELoss(y, y_hat,l_norm):
#     mask = divide_no_nan(mask, t.abs(y))
    mape = torch.abs((y - y_hat)/y)
    mape = torch.mean(mape)
    return mape/l_norm


# In[14]:


data,rel= next(iter(test_loader))
print(data[:,:,:,0:2].size())


# In[15]:


a = torch.randn(4, 4,2)
print(torch.mean(torch.sum(a,1)/4))
torch.mean(a)


# ### **Training**

# In[16]:


# train_losses = np.empty(T * epochs, dtype=float)

def train(epoch, best_val_loss):
    t = time.time()
    nll_train = []
    acc_train = []
    kl_train = []
    mse_train = []
    mae_train = []
    mape_train = []

    output_var=1e-3
    model.train()
    for batch_idx, (data, relations) in enumerate(train_loader):
        # rel_rec = Variable(torch.from_numpy(adjacency[d]))
        # rel_send = Variable(torch.from_numpy(adjacency[d]))
        optimizer.zero_grad()
        # data = data[:,:,0:-1,:]
        target = data[:,:,1:,:]
        data, relations = data.to(device), relations.to(device)
        target= target.to(device)
        # x = data.x.unsqueeze(0).unsqueeze(2)
        x_recon, logits, edges = model(data, rel_rec, rel_send, pred_steps=1)
        prob = my_softmax(logits, -1)
        loss_nll = nll_gaussian(x_recon, target, output_var)

        loss_kl = kl_categorical_uniform(prob, num_atoms, 2)
        loss_norm =LossNorm(data)
#         print(loss_norm)
        loss = loss_nll + loss_kl
        mae_loss =MAELoss(target[:,:,:,0:2], x_recon[:,:,:,0:2], loss_norm).detach().cpu().numpy()
        mape_loss =MAPELoss(target[:,:,:,0:2], x_recon[:,:,:,0:2],loss_norm).detach().cpu().numpy()
        acc = edge_accuracy(logits, relations)
        acc_train.append(acc)
        mae_train.append(mae_loss)
        mape_train.append(mape_loss)
        loss.backward()
        optimizer.step()

        mse_train.append((F.mse_loss(x_recon[:,:,:,0:2], target[:,:,:,0:2])/loss_norm).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
    if(epoch % 20 == 0):
        print('Epoch: {:04d}'.format(epoch),
          'nll_train: {:.4f}'.format(np.mean(nll_train)),
          'kl_train: {:.4f}'.format(np.mean(kl_train)),
          'mse_train: {:.4f}'.format(np.mean(mse_train)),
          'mae_train: {:.4f}'.format(np.mean(mae_train)),
          'mape_train: {:.4f}'.format(np.mean(mape_train)),
          'acc_train: {:.4f}'.format(np.mean(acc_train)))
    return np.mean(nll_train),np.mean(kl_train),np.mean(mse_train),np.mean(mae_train),np.mean(mape_train),np.mean(acc_train)
    
def test(epoch, best_val_loss,loader):
    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []
    mae_val = []
    mape_val = []
    pred_output=[]
    output_var=1e-3

    model.eval()
    for batch_idx, (data, relations) in enumerate(valid_loader):

        target = data[:,:,1:,:]
        data, relations = data.to(device), relations.to(device)
        target= target.to(device)
        # x = data.x.unsqueeze(0).unsqueeze(2)
        x_recon, logits, edges = model(data, rel_rec, rel_send,pred_steps=1)
        prob = my_softmax(logits, -1)
        # print(data.size(),'data input')
#         print(x_recon.size(), 'x recon')
#         pred_x = x_recon.squeeze().detach().cpu().numpy()
#         pred_x = torch.cat(x_recon)
        loss_nll = nll_gaussian(x_recon, target, output_var)


        loss_kl = kl_categorical_uniform(prob, num_atoms, 2)
        loss_norm =LossNorm(target)

        acc = edge_accuracy(logits, relations)
        mae_loss =MAELoss(target[:,:,:,0:2], x_recon[:,:,:,0:2], loss_norm).detach().cpu().numpy()
        mape_loss =MAPELoss(target[:,:,:,0:2], x_recon[:,:,:,0:2], loss_norm).detach().cpu().numpy()
        mae_val.append(mae_loss)
        mape_val.append(mape_loss)
        acc_val.append(acc)

        mse_val.append((F.mse_loss(x_recon[:,:,:,0:2], target[:,:,:,0:2])/loss_norm).item())
        nll_val.append(loss_nll.item())
        kl_val.append(loss_kl.item())
#     print(np.shape(pred_output))
    if(epoch % 20 == 0):

        print('Epoch: {:04d}'.format(epoch),
          'nll_val: {:.4f}'.format(np.mean(nll_val)),
          'kl_val: {:.4f}'.format(np.mean(kl_val)),
          'mse_val: {:.4f}'.format(np.mean(mse_val)),
          'mae_val: {:.4f}'.format(np.mean(mae_val)),
          'mape_val: {:.4f}'.format(np.mean(mape_val)),
          'acc_val: {:.4f}'.format(np.mean(acc_val)))
#           'time: {:.4f}s'.format(time.time() - t))
    return pred_output, np.mean(nll_val),np.mean(kl_val),np.mean(mse_val),np.mean(mae_val),np.mean(mape_val),np.mean(acc_val)
nepochs=300
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0
nll_val_loses = []
acc_val_loses = []
kl_val_loses = []
mae_train_loses = []
mape_train_loses = []
mse_val_loses = []
nll_train_loses = []
acc_train_loses = []
kl_train_loses = []
mse_train_loses = []
mae_val_loses=[]
mape_val_loses=[]
for epoch in range(nepochs):
    nll, kl, mse,mae,mape, acc = train(epoch, best_val_loss)# 
    _,nll_val, kl_val, mse_val,mae_val, mape_val, acc_val = test(epoch, best_val_loss,valid_loader)# 

    nll_train_loses.append(nll)
    acc_train_loses.append(acc)
    kl_train_loses.append(kl)
    mse_train_loses.append(mse)
    mae_train_loses.append(mae)
    mape_train_loses.append(mape)
    mae_val_loses.append(mae_val)
    mape_val_loses.append(mape_val)
    nll_val_loses.append(nll_val)
    kl_val_loses.append(kl_val)
    mse_val_loses.append(mse_val)
    acc_val_loses.append(acc_val)


# In[17]:


plt.plot(mse_train_loses,'b')
plt.title("MSE loses(train)")
plt.ylabel("MSE")
plt.xlabel("epochs")
plt.grid(color="whitesmoke")
plt.savefig('images_boids/'+'msetrain'+'boids',dpi=300)
# plt.ylabels("")

plt.figure()
plt.plot(acc_train_loses,'b')
plt.title("Edges reconstruction Accuracy(train)")
plt.xlabel("epochs")
plt.grid(color="whitesmoke")

plt.ylabel("reconstruction accuracy")
plt.savefig('images_boids/'+'accuracytrain'+'boids',dpi=300)

plt.figure()
plt.plot([-1*kl for kl in kl_train_loses],'b')
plt.title("KL divergence(train))")
plt.ylabel("KL divergence")
plt.xlabel("epochs")
plt.grid(color="whitesmoke")

plt.savefig('images_boids/'+'KL_divertrain'+'boids',dpi=300)

plt.figure()
plt.plot(mae_train_loses,'b')
plt.title("MAE loses(train)")
plt.xlabel("epochs")
plt.ylabel("MAE loss (train)")
plt.grid(color="whitesmoke")
plt.savefig('images_boids/'+'maetrain'+'boids',dpi=300)

# plt.figure()
# plt.plot(mape_train_loses)
# plt.title("MAPE loses(train)")
# plt.xlabel("epochs")
# plt.savefig('images_boids/'+'mapetrain'+'boids',dpi=300)

plt.figure()
plt.plot(mae_val_loses,'b')
plt.title("MAE loses(validation)")
plt.xlabel("epochs")
plt.grid(color="whitesmoke")
plt.ylabel("MAE loss validation")
plt.savefig('images_boids/'+'maevalloss'+'boids',dpi=300)

# plt.figure()
# plt.plot(mape_val_loses)
# plt.title("MAPE loses(validation)")
# plt.xlabel("epochs")
# plt.savefig('images_boids/'+'mapeloss'+'boids',dpi=300)

plt.figure()
plt.plot(mse_val_loses,'b')
plt.title("MSE loses(Validation)")
plt.xlabel("epochs")
plt.grid(color="whitesmoke")
plt.ylabel("MSE loss validation")
plt.savefig('images_boids/'+'mseval'+'boids',dpi=300)


plt.figure()
plt.plot(acc_val_loses,'b')
plt.title("Edges reconstruction Accuracy(Validation)")
plt.xlabel("epochs")
plt.grid(color="whitesmoke")
plt.ylabel("reconstruction accuracy")
plt.savefig('images_boids/'+'accuracyvalidation'+'boids',dpi=300)

plt.figure()
plt.plot([-1*kl for kl in kl_val_loses],'b')
plt.title("KL divergence(Validation))")
plt.xlabel("epochs")
plt.grid(color="whitesmoke")
plt.savefig('images_boids/'+'kldiver_validation'+'boids',dpi=300)


# In[26]:


# del pred
# del g_truth
output_var=1e-3

        
        
def test_model(loader, step):
#     pred_output=[]
    nll_test = []
    acc_test = []
    kl_test = []
    mse_test = []
    mae_test=[]
    mape_test=[]
    model.eval()
    for batch_idx, (data, relations) in enumerate(test_loader):
        target = data[:,:,1:,:]
        data, relations = data.to(device), relations.to(device)
        target= target.to(device)
        # x = data.x.unsqueeze(0).unsqueeze(2)
        x_recon, logits, edges = model(data, rel_rec, rel_send,pred_steps=step)
        prob = my_softmax(logits, -1)

        loss_nll = nll_gaussian(x_recon, target, output_var)


        loss_kl = kl_categorical_uniform(prob, num_atoms, 2)
        loss_norm =LossNorm(data)
        acc = edge_accuracy(logits, relations)
        acc_test.append(acc)
        mae_loss =MAELoss(target, x_recon, loss_norm).detach().cpu().numpy()
        mape_loss =MAPELoss(target, x_recon, loss_norm).detach().cpu().numpy()
        
        mse_test.append((F.mse_loss(x_recon, target)/loss_norm).item())
        mae_test.append(mae_loss)
        mape_test.append(mape_loss)
        nll_test.append(loss_nll.item())
        kl_test.append(loss_kl.item())
    #     print(np.shape(pred_output))
        try:
            pred = torch.cat((pred, x_recon), dim=0)
        except:
            pred = x_recon
        try:
            g_truth = torch.cat((g_truth, target), dim=0)
        except:
            g_truth = target
    mse_loss = np.mean(mse_test)
    mae_loss = np.mean(mae_test)
    mape_loss = np.mean(mape_test)
    predi=pred.squeeze().detach().cpu().numpy().transpose(1,0,2,3).reshape(10,-1,4)
    g_truth =g_truth.squeeze().detach().cpu().numpy().transpose(1,0,2,3).reshape(10,-1,4)
    return mse_loss, mae_loss, mape_loss, predi, g_truth


# In[28]:


step=5
mse_loss, mae_loss, mape_loss, predi, g_truth=test_model(test_loader, step)
print(mse_loss)
print(mae_loss)
# def normalize(data, data_max, data_min):
#     return (data - data_min) * 2 / (data_max - data_min) - 1

# # loc_max, loc_min, vel_max, vel_min
# def urnnormalize(data, data_max, data_min):
#     return (data + 1) * (data_max - data_min) / 2. + data_min

# position_unnormalize = urnnormalize(predi[:,:,0:2], loc_max, loc_min)
# velocity_unnormalize = urnnormalize(predi[:,:,2:4], vel_max, vel_min)
# # target

# gt_position_unnormalize = urnnormalize(g_truth[:,:,0:2], loc_max, loc_min)
# gt_velocity_unnormalize = urnnormalize(g_truth[:,:,2:4], vel_max, vel_min)

# predicted_features = np.concatenate([position_unnormalize, velocity_unnormalize],axis=2)
# #     print(predicted_features.shape)
# target_features =np.concatenate([gt_position_unnormalize, gt_velocity_unnormalize ],axis=2)
# #     print(target_features.shape)


# ## Calculate Metrics: AVD and AMD

# In[20]:


def metric_amd(predict):
    N,T,D=np.shape(predict)
    amd = 0
    adv= 0
    for j in range(2):
        for i in range(10):
            predi_neigh= np.delete(predict, i, 0)
#             print(np.shape(predi_neigh))
            dist_diff=np.abs(predict[i,:,j]-predi_neigh[:,:,j])
            amd=amd+(np.min(dist_diff, axis=0))
#             print(amd)
    return amd/(N)

def metric_avd(predict):
    N,T,D=np.shape(predict)
    amd = 0
    adv= 0
    for j in range(2,4):
        for i in range(10):
            predi_neigh= np.delete(predict, i, 0)
            dist_diff=np.abs(predict[i,:,j]-predi_neigh[:,:,j])
            amd=amd+(np.sum(dist_diff, axis=0))
#             print(amd)
    return 2*amd/(N*(N-1))


amd_total_predi=metric_amd(predicted_features)
amd_total_gtruth=metric_amd(target_features)



avd_total_predi=metric_avd(predicted_features)
avd_total_gtruth=metric_avd(target_features)


amd_total = amd_total_predi.mean()
avd_total = avd_total_predi.mean()

amd_total_gt = amd_total_gtruth.mean()
avd_total_gt = avd_total_gtruth.mean()


print(amd_total,"amd total")
print(avd_total,"avd total")

print(amd_total_gt,"amd total gt")
print(avd_total_gt,"avd total gt")


# print(amd_total)
# print(np.shape(avd_total))
plt.plot(amd_total_predi, label="Prediction",c='orangered')
plt.plot(amd_total_gtruth,label="Ground truth",c='dodgerblue')
plt.xlabel("timestep")
plt.ylabel("AMD")
plt.legend()
plt.figure()
plt.plot(avd_total_predi, label="Prediction",c='orangered')
plt.plot(avd_total_gtruth,label="Ground truth",c='dodgerblue')
plt.xlabel("timestep")
plt.ylabel("AVD")
plt.legend()


# In[21]:




def plot_result(predi, g_truth, step):
    output = predi
    positions = output[:,:,0:2]
    velocity = output[:,:,2:4]
    def normalize(data, data_max, data_min):
        return (data - data_min) * 2 / (data_max - data_min) - 1

    # loc_max, loc_min, vel_max, vel_min
    def urnnormalize(data, data_max, data_min):
        return (data + 1) * (data_max - data_min) / 2. + data_min

    position_unnormalize = urnnormalize(positions, loc_max, loc_min)
    velocity_unnormalize = urnnormalize(velocity, vel_max, vel_min)
    # target

    gt_position_unnormalize = urnnormalize(g_truth[:,:,0:2], loc_max, loc_min)
    gt_velocity_unnormalize = urnnormalize(g_truth[:,:,2:4], vel_max, vel_min)

    predicted_features = np.concatenate([position_unnormalize, velocity_unnormalize],axis=2)
#     print(predicted_features.shape)
    target_features =np.concatenate([gt_position_unnormalize, gt_velocity_unnormalize ],axis=2)
#     print(target_features.shape)
#     plt.figure(figsize=(6, 6))
#     G = gridspec.GridSpec(3, 4)
#     time=[5, 100, 500, 1000, 1500]
#     tpast=10
#     for t in range(10, 990,50):
    plt.figure()
    for i in range(10):
            plt.plot(predicted_features[i,:300,0],predicted_features[i,:300,1])
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.figure()
    for i in range(10):
#             plt.plot(predicted_features[i,:,0],predicted_features[i,:,1])
            plt.plot(target_features[i,:300,0],target_features[i,:300,1])
    plt.xlabel("x position")
    plt.ylabel("y position")

step=1
mse_loss, mae_Loss, mape_loss, predi, g_truth=test_model(test_loader, step)
plot_result(predi, g_truth, step=step)


# In[23]:




def plot_result(predi, g_truth, step):
    output = predi
    positions = output[:,:,0:2]
    velocity = output[:,:,2:4]
    def normalize(data, data_max, data_min):
        return (data - data_min) * 2 / (data_max - data_min) - 1

    # loc_max, loc_min, vel_max, vel_min
    def urnnormalize(data, data_max, data_min):
        return (data + 1) * (data_max - data_min) / 2. + data_min

    position_unnormalize = urnnormalize(positions, loc_max, loc_min)
    velocity_unnormalize = urnnormalize(velocity, vel_max, vel_min)
    # target

    gt_position_unnormalize = urnnormalize(g_truth[:,:,0:2], loc_max, loc_min)
    gt_velocity_unnormalize = urnnormalize(g_truth[:,:,2:4], vel_max, vel_min)

    predicted_features = np.concatenate([position_unnormalize, velocity_unnormalize],axis=2)
#     print(predicted_features.shape)
    target_features =np.concatenate([gt_position_unnormalize, gt_velocity_unnormalize ],axis=2)
#     print(target_features.shape)
#     plt.figure(figsize=(6, 6))
#     G = gridspec.GridSpec(3, 4)
#     time=[5, 100, 500, 1000, 1500]
    tpast=2
    for t in range(0, 999,100):
        plt.figure()
        for i in range(10):
            plt.plot(predicted_features[i,t-tpast:t,0],predicted_features[i,t-tpast:t,1],c='orangered')
            plt.plot(target_features[i,t-tpast:t,0],target_features[i,t-tpast:t,1],c='dodgerblue')
            plt.scatter(predicted_features[i,t-1,0],predicted_features[i,t-1,1],c='orangered')
            plt.scatter(target_features[i,t-1,0],target_features[i,t-1,1],c='dodgerblue')
            title="t="+str(t)
            plt.title(title)
            plt.legend(["Ground Truth","Predicted"])
            plt.savefig('newimages/'+'boidst_'+str(t),dpi=300)


        #         plt.plot(predicted_features[1,:5,0],predicted_features[1,:5,1])
#         plt.plot(predicted_features[2,:5,0],predicted_features[2,:5,1])
#         plt.plot(predicted_features[3,:5,0],predicted_features[3,:5,1])
#         plt.plot(predicted_features[4,:5,0],predicted_features[4,:5,1])

step=1
mse_loss, mae_Loss, mape_loss, predi, g_truth=test_model(test_loader, step)
plot_result(predi, g_truth, step=step)


# In[ ]:




def plot_result(predi, g_truth, step):
    output = predi
    positions = output[:,:,0:2]
    velocity = output[:,:,2:4]
    def normalize(data, data_max, data_min):
        return (data - data_min) * 2 / (data_max - data_min) - 1

    # loc_max, loc_min, vel_max, vel_min
    def urnnormalize(data, data_max, data_min):
        return (data + 1) * (data_max - data_min) / 2. + data_min

    position_unnormalize = urnnormalize(positions, loc_max, loc_min)
    velocity_unnormalize = urnnormalize(velocity, vel_max, vel_min)
    # target

    gt_position_unnormalize = urnnormalize(g_truth[:,:,0:2], loc_max, loc_min)
    gt_velocity_unnormalize = urnnormalize(g_truth[:,:,2:4], vel_max, vel_min)

    predicted_features = np.concatenate([position_unnormalize, velocity_unnormalize],axis=2)
#     print(predicted_features.shape)
    target_features =np.concatenate([gt_position_unnormalize, gt_velocity_unnormalize ],axis=2)
#     print(target_features.shape)
#     plt.figure(figsize=(8, 9))
#     G = gridspec.GridSpec(3, 4)
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(7, 9))
    axes = axes.flatten()
    for i in range(12):
        if i<=9:
            axes[i].plot(predicted_features[i-1,:,0],predicted_features[i-1,:,1],'b')
            axes[i].plot(target_features[i-1,:,0],target_features[i-1,:,1],  'r--')
            axes[i].set_title("agent "+str(i+1),fontsize = 8)
            axes[i].tick_params(axis='x', labelsize=8)
            axes[i].tick_params(axis='y', labelsize=8)

        if i == 9:
            axes[i].legend(['predicted','ground truth'],fontsize=8)
    fig.text(0.5, 0.04, 'position x(cm)', ha='center')
    fig.text(0.0, 0.5, 'position y (cm)', va='center', rotation='vertical')
    axes[-1].axis('off')
    axes[-2].axis('off')
    plt.tight_layout()
    plt.savefig('images_boids/'+'agents_traj'+str(step),dpi=300)

step=1
mse_loss, mae_Loss, mape_loss, predi, g_truth=test_model(test_loader, step)
plot_result(predi, g_truth, step=step)


# In[ ]:




def plot_1result(predi, g_truth, step):
    output = predi
    positions = output[:,:,0:2]
    velocity = output[:,:,2:4]
    def normalize(data, data_max, data_min):
        return (data - data_min) * 2 / (data_max - data_min) - 1

    # loc_max, loc_min, vel_max, vel_min
    def urnnormalize(data, data_max, data_min):
        return (data + 1) * (data_max - data_min) / 2. + data_min

    position_unnormalize = urnnormalize(positions, loc_max, loc_min)
    velocity_unnormalize = urnnormalize(velocity, vel_max, vel_min)
    # target

    gt_position_unnormalize = urnnormalize(g_truth[:,:,0:2], loc_max, loc_min)
    gt_velocity_unnormalize = urnnormalize(g_truth[:,:,2:4], vel_max, vel_min)

    predicted_features = np.concatenate([position_unnormalize, velocity_unnormalize],axis=2)
#     print(predicted_features.shape)
    target_features =np.concatenate([gt_position_unnormalize, gt_velocity_unnormalize ],axis=2)
#     print(target_features.shape)
#     plt.figure(figsize=(8, 9))
#     G = gridspec.GridSpec(3, 4)
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(7, 9))
    axes = axes.flatten()
    for i in range(12):
        if i<=9:
            axes[i].plot(predicted_features[i-1,:,0],predicted_features[i-1,:,1],'b')
            axes[i].plot(target_features[i-1,:,0],target_features[i-1,:,1],  'r--')
            axes[i].set_title("agent "+str(i+1),fontsize = 8)
            axes[i].tick_params(axis='x', labelsize=8)
            axes[i].tick_params(axis='y', labelsize=8)

        if i == 9:
            axes[i].legend(['predicted','ground truth'],fontsize=8)
    fig.text(0.5, 0.04, 'position x(cm)', ha='center')
    fig.text(0.0, 0.5, 'position y (cm)', va='center', rotation='vertical')
    axes[-1].axis('off')
    axes[-2].axis('off')
    plt.tight_layout()
    plt.savefig('images_boids/'+'agents_traj'+str(step),dpi=300)

step=1
mse_loss, mae_Loss, mape_loss, predi, g_truth=test_model(test_loader, step)
plot_1result(predi, g_truth, step=step)


# In[ ]:


# for step in range(1,26):
#     mse_loss, mae_Loss, mape_loss, predi, g_truth=test_model(test_loader, step)
#     plot_result(predi, g_truth, step=step)


# In[ ]:


def plot_result(predi, g_truth, step):
    output = predi
    positions = output[:,:,0:2]
    velocity = output[:,:,2:4]
    def normalize(data, data_max, data_min):
        return (data - data_min) * 2 / (data_max - data_min) - 1

    # loc_max, loc_min, vel_max, vel_min
    def urnnormalize(data, data_max, data_min):
        return (data + 1) * (data_max - data_min) / 2. + data_min

    position_unnormalize = urnnormalize(positions, loc_max, loc_min)
    velocity_unnormalize = urnnormalize(velocity, vel_max, vel_min)
    # target

    gt_position_unnormalize = urnnormalize(g_truth[:,:,0:2], loc_max, loc_min)
    gt_velocity_unnormalize = urnnormalize(g_truth[:,:,2:4], vel_max, vel_min)

    predicted_features = np.concatenate([position_unnormalize, velocity_unnormalize],axis=2)
#     print(predicted_features.shape)
    target_features =np.concatenate([gt_position_unnormalize, gt_velocity_unnormalize ],axis=2)
#     print(target_features.shape)

#     plt.figure(figsize=(15, 15))

    fig, (ax1, ax2)  = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(predicted_features[0,:,0],predicted_features[0,:,1],'b',label='prediction')
    ax1.plot(target_features[0,:,0],target_features[0,:,1],  'r--',label='ground truth')
    ax1.legend()
    ax2.plot(abs(predicted_features[0,:,0]-target_features[0,:,0]), label='pos(x)')
    ax2.plot(abs(predicted_features[0,:,1]-target_features[0,:,1]), label ='pos(y)')
    title1= "trajectory plot for prediction steps="+str(step)
    title2= "absolute error for prediction steps="+str(step)
    ax2.legend()
    ax1.title.set_text(title1)
    ax2.title.set_text(title2)
    plt.savefig('images_boids/'+'traj_error'+str(step)+'png',dpi=300)
step=5
mse_loss, mae_Loss, mape_loss, predi, g_truth=test_model(test_loader, step)
plot_result(predi, g_truth, step=step)


# In[ ]:


mae=[]
mape=[]
mse=[]
for step in range(1,26):
    mse_loss, mae_loss, mape_loss, predi, g_truth=test_model(test_loader, step)
    mae.append(mae_loss)
    mape.append(mape_loss)
    mse.append(mse_loss)
#     plot_result(predi, g_truth, step=step)
    if (step % 5 ==0):
        print(mse_loss)


# In[ ]:


plt.figure()
plt.plot(mae,'b')
dim = np.arange(1,26,2);
plt.xticks(dim)
plt.xlim([1, 25])
plt.title("MAE loss per prediction step")
plt.xlabel("K step ahead")
plt.ylabel("MAE")
plt.grid(color="whitesmoke")
plt.savefig('images_boids/'+'mae_test_boids',dpi=300)

# plt.figure()
# plt.plot(mape,'b')
# plt.title("MAPE loss per prediction step")
# plt.xlabel("prediction step")
# plt.savefig('images/'+'mape'+'png',dpi=300)

plt.figure()
plt.plot(mse,'b')
plt.xlim([1, 25])
dim = np.arange(1,26,2);
plt.xticks(dim)
plt.title("MSE loss per epoch")
plt.xlabel("K step ahead")
plt.ylabel("MSE")
plt.grid(color="whitesmoke")
plt.savefig('images_boids/'+'mse_test_boids',dpi=300)


# In[ ]:


# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.colors import Normalize
# mse_loss, mae_Loss, mape_loss, predicted_features,  target_features=test_model(test_loader, step=5)
# np.save('images_boids/'+'prediction_data' + '.npy', predicted_features)
# np.save('images_boids/'+'target_data' + '.npy', target_features)

# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']

# for i, color in enumerate(colors):

#     # assume the arrays for the two robots are called 'robot1_data' and 'robot2_data'
#     x_positions_1 = predicted_features[i, :,0]
#     y_positions_1 =predicted_features[i, :,1]
#     times_1 = np.arange(len(x_positions_1)) # create an array of time stamps
#     norm_1 = Normalize(vmin=0, vmax=len(x_positions_1)) # normalize time stamps to [0,1]

#     x_positions_2 = target_features[i, :,0]
#     y_positions_2 = target_features[i, :,1]
#     times_2 = np.arange(len(x_positions_2)) # create an array of time stamps
#     norm_2 = Normalize(vmin=0, vmax=len(x_positions_2)) # normalize time stamps to [0,1]
# #     color1 = positions[:,:,1]

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

#     ax1.scatter(x_positions_1, y_positions_1, c='red',alpha=0.2)
#     ax1.set_xlabel('X Position')
#     ax1.set_ylabel('Y Position')
#     ax1.set_title('Prediction')

#     sc2 = ax2.scatter(x_positions_2, y_positions_2, c='blue', alpha=0.2)
#     ax2.set_xlabel('X Position')
#     ax2.set_ylabel('Y Position')
#     ax2.set_title('Truth')

#     plt.tight_layout()
#     plt.show()


# In[ ]:


for batch_idx, (data, relations) in enumerate(val_loader):
    model.eval()
    output, logits, edges = model(data.to(device), rel_rec, rel_send)

    output_np=output.detach().cpu().numpy()


# In[ ]:



import itertools
import itertools
from collections import deque
from IPython import display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
agentPositions= predicted_features
dataX= agentPositions[:,:,0]
dataY= agentPositions[:,:,1]
xmin = dataX.min()
ymin = dataY.min()
xmax = dataX.max()
ymax = dataY.max()
# data1 = np.load(data/edges_test_springs5.npy)
def animate_func(i):
    ax.clear()  # Clears the figure to update the line, point,   

    ax.scatter(dataX[:,i], dataY[:,i], c='red', marker='o',label='prediction')
    ax.set_xlim(xmin-0.5, xmax+0.5)
    ax.set_ylim(ymin-0.5, ymax+0.5)
    ax.legend()
# numDataPoints=49
fig, ax = plt.subplots()
line_ani = animation.FuncAnimation(fig, animate_func, interval=20,   
                                   frames=np.shape(dataX)[1])
# plt.show()

line_ani.save('images_boids/'+'boids_pred_tracj.gif', writer='pillow')


# In[ ]:



import itertools
import itertools
from collections import deque
from IPython import display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
dataX= target_features[:,:,0]
dataY= target_features[:,:,1]
xmin = dataX.min()
ymin = dataY.min()
xmax = dataX.max()
ymax = dataY.max()
# data1 = np.load(data/edges_test_springs5.npy)
def animate_func(i):
    ax.clear()  # Clears the figure to update the line, point,   

    ax.scatter(dataX[:,i], dataY[:,i], c='blue', marker='o',label="target")
    ax.set_xlim(xmin-0.5, xmax+0.5)
    ax.set_ylim(ymin-0.5, ymax+0.5)
    ax.legend()

# numDataPoints=49
fig, ax = plt.subplots()
line_ani = animation.FuncAnimation(fig, animate_func, interval=20,   
                                   frames=np.shape(dataX)[1])
# plt.show()

line_ani.save('images_boids/'+'boid_target_tracj.gif', writer='pillow')


# In[ ]:



import itertools
import itertools
from collections import deque
from IPython import display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
# agentPositions= target_features
targetX= target_features[:,:,0]
targetY= target_features[:,:,1]

predictedX= predicted_features[:,:,0]
predictedY= predicted_features[:,:,1]

xmin = dataX.min()
ymin = dataY.min()
xmax = dataX.max()
ymax = dataY.max()
# data1 = np.load(data/edges_test_springs5.npy)
def animate_func(i):
    ax.clear()  # Clears the figure to update the line, point,   

    ax.scatter(targetX[:,i], targetY[:,i], c='blue', marker='.', label="target")
    ax.scatter(predictedX[:,i], predictedY[:,i], c='red', marker='.',label="prediction")

    ax.set_xlim(xmin-0.5, xmax+0.5)
    ax.set_ylim(ymin-0.5, ymax+0.5)
    ax.legend()
# numDataPoints=49
fig, ax = plt.subplots()
line_ani = animation.FuncAnimation(fig, animate_func, interval=20,   
                                   frames=np.shape(dataX)[1])
# plt.show()

line_ani.save('images_boids/'+'boids_pred_target_tracj.gif', writer='pillow')


# In[ ]:


print(np.shape(predicted_features))

