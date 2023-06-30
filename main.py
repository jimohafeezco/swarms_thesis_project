#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sklearn.metrics as metrics

import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.datasets import KarateClub

import os.path as osp

import numpy as np
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_geometric.utils import to_torch_coo_tensor
# from torch_geometric.utils import to_edge_index
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj,to_networkx
from torch_geometric.nn.models import basic_gnn
from torch.nn import Linear
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn import Set2Set, BatchNorm
from torch_geometric_temporal.nn.recurrent import GConvGRU

from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import global_mean_pool, global_add_pool

from torch_geometric.nn import GAE, VGAE, GCNConv,GCNConv, GINConv
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
# import npzviewer

from torch.utils.data.dataset import TensorDataset


# In[2]:


dir = 'swarm_data/data/'
filename = 'group_1_fish'

names = [filename+str(num) +'.npz' for num in range(0,10)]
print(names)


# In[3]:


def urnnormalize(data, data_max, data_min):
	return (data + 1) * (data_max - data_min) / 2. + data_min

def load_motion_data(batch_size=32):
    features = np.load('features_data1' + '.npy')
    edges =  np.load('edges_data1' + '.npy')

    loc_max = features[:,:,0:2].max()
    loc_min = features[:,:,0:2].min()
    vel_max = features[:,:,2:4].max()
    vel_min = features[:,:,2:4].min()

    # Normalize to [-1, 1]
    loc_train = (features[:,:,0:2] - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (features[:,:,2:4]- vel_min) * 2 / (vel_max - vel_min) - 1
    features =np.concatenate([loc_train, vel_train], axis=2)
    
    return features, edges,loc_max, loc_min, vel_max, vel_min

features, edges,loc_max, loc_min, vel_max, vel_min= load_motion_data(batch_size=32)
features_tensor= torch.FloatTensor(features)
edges_tensor = torch.FloatTensor(edges)

train_tensor = TensorDataset(features_tensor, edges_tensor)
print(np.shape(features),'features')
print(np.shape(edges),'edges')


# In[4]:


data_list = []
N=5000
for i in range(N-1):
  edge_list = dense_to_sparse(edges_tensor[i])
  # print(edge_list)
  data=Data(x=features_tensor[i], edge_index=edge_list[0], y = features_tensor[i+1])
  data_list.append(data)
loader = DataLoader(data_list, batch_size=64)
dataset= data_list
edge_adj=to_dense_adj(dataset[0].edge_index)
print(edge_adj[0])
triu_indices = torch.triu_indices(10,10, offset=1)

print(triu_indices)
triu_mask=torch.squeeze(to_dense_adj(triu_indices)).bool()
print(triu_mask)
triu_logits=edge_adj[0][triu_mask].type(torch.int) 
print(triu_logits)


# In[5]:



def triu_to_dense(triu_values, num_nodes):
    dense_adj = torch.zeros((num_nodes, num_nodes)).int()
    triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
    tril_indices = torch.tril_indices(num_nodes, num_nodes, offset=-1)
    dense_adj[triu_indices[0], triu_indices[1]] = triu_values
    dense_adj[tril_indices[0], tril_indices[1]] = triu_values
    return dense_adj
dense_from_triu=triu_to_dense(triu_logits,10)
print(dense_from_triu)


# In[6]:


from torch_geometric.loader import DataLoader

# Create training, validation, and test sets
train_dataset = dataset[:int(len(dataset)*0.8)]
val_dataset   = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
test_dataset  = dataset[int(len(dataset)*0.9):]
# print(test_dataset)
print(f'Training set   = {len(train_dataset)} graphs')
print(f'Validation set = {len(val_dataset)} graphs')
print(f'Test set       = {len(test_dataset)} graphs')

batch_size= 64
# Create mini-batches
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# print('\nTrain loader:')
# for i, subgraph in enumerate(train_loader):
#     print(f' - Subgraph {i}: {subgraph}')

# print('\nValidation loader:')
# for i, subgraph in enumerate(val_loader):
#     print(f' - Subgraph {i}: {subgraph}')

# print('\nTest loader:')
# for i, subgraph in enumerate(test_loader):
#     print(f' - Subgraph {i}: {subgraph}')


# In[7]:



class VGAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        feature_size = 4
        out_channels=4
        encoder_embedding_size=32
        latent_size=4
        decoder_embedding_size=64
        self.recurrent = GConvGRU(feature_size, encoder_embedding_size*2,1)
#         self.conv1 = GCNConv(feature_size, encoder_embedding_size)
        self.conv1 = GCNConv(encoder_embedding_size*2, encoder_embedding_size*2)
        self.conv2 = GCNConv(encoder_embedding_size*2, encoder_embedding_size)

#         self.bn1 = BatchNorm(encoder_embedding_size*2)
#         self.bn2 = BatchNorm(encoder_embedding_size)

        self.conv_mean = GCNConv(encoder_embedding_size, latent_size)
        self.conv_logvar = GCNConv(encoder_embedding_size, latent_size)

        self.linear1 = torch.nn.Linear(latent_size, decoder_embedding_size*2)
        self.linear2 = torch.nn.Linear(decoder_embedding_size*2, decoder_embedding_size)
        self.linear3 = torch.nn.Linear(decoder_embedding_size, out_channels)

        self.conv1_d = GCNConv(latent_size, decoder_embedding_size)
        self.conv2_d = GCNConv(decoder_embedding_size, feature_size)

        self.weight = Parameter(torch.Tensor(feature_size, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)

    def encoder(self, x, edge_index,batch_index):

        x = self.recurrent(x, edge_index).relu()
#         x = self.bn1(x)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
    #         x = self.bn2(x)

        mu = self.conv_mean(x, edge_index).relu()

        logvar = self.conv_logvar(x, edge_index).relu()
        
        z = self.reparametrize(mu, logvar)
        return mu,logvar, z
        
    def decode_edges(self, z: Tensor, sigmoid: bool = True) -> Tensor:

        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj
    
    def reparametrize(self, mu: Tensor, logstd: Tensor) -> Tensor:
        return mu + torch.randn_like(logstd) * torch.exp(logstd)
    def kl_loss(self, mu: Tensor = None,
                logstd: Tensor = None) -> Tensor:
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
        
#     def decode_edges(self, z: Tensor, sigmoid: bool = True) -> Tensor:

#         adj = torch.matmul(z, z.t())
#         return torch.sigmoid(adj) if sigmoid else adj

#     def decoder_conv(self, z, edge_index):
#         x = F.relu(self.conv1_d(z, edge_index))
#         x = self.conv2_d(x, edge_index)
#         return x
    def decode_graph(self, graph_z):  
        z = self.linear1(graph_z).relu()
        z = self.linear2(z).relu()
        x = self.linear3(z)
        return x

    def decoder(self, z,batch_index):
        node_features =[]
        edges_logits = []
        for graph_id in torch.unique(batch_index):
          graph_mask = torch.eq(batch_index, graph_id)
#           print(graph_mask)
          graph_z = z[graph_mask]
#           print(graph_z,'graph z')
          node= self.decode_graph(graph_z)
          # print(node.size(),'node size')
          node_features.append(node)
          decode_edges = self.decode_edges(graph_z)
#           print(decode_edges)  
          edges_logits.append(decode_edges)
#         node_features =torch.cat(node_features)
        edges_logits = torch.cat(edges_logits)
        # print(node_features,'node')
        return edges_logits, node_features

    def forward(self, x, adj,batch_index):
        mu, logvar, embeddings= self.encoder(x, adj,batch_index)
        edges_adj ,nf= self.decoder(embeddings, batch_index)
        return mu, logvar, embeddings, edges_adj

model= VGAE()
model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
model


# In[8]:


import time
def nll_gaussian(preds, target, var, add_const=False):
    # neg_log_p = ((preds - target) ** 2 / (2 * variance))
    var_tensor = torch.ones(preds.size(),requires_grad=True).to(device) *var
    loss = 0.5 * (torch.log(var_tensor) + (preds - target)**2 / var)

    return loss.mean()
def slice_graph_targets(graph_id, batch_targets, batch_index):

    # Create mask for nodes of this graph id
    graph_mask = torch.eq(batch_index, graph_id)

    graph_targets = batch_targets[graph_mask][:, graph_mask]
    # Get triangular upper part of adjacency matrix for targets
#     print(graph_targets.size())
    triu_indices = torch.triu_indices(graph_targets.shape[0], graph_targets.shape[0], offset=1)
    triu_mask = torch.squeeze(to_dense_adj(triu_indices)).bool()
    graph_targets_triu= graph_targets[triu_mask]
    return graph_targets_triu
def slice_graph_pred(graph_id, batch_pred, batch_index):

    # Create mask for nodes of this graph id
    graph_mask = torch.eq(batch_index, graph_id)
    graph_pred = batch_pred[graph_mask]

    triu_indices = torch.triu_indices(graph_pred.shape[0], graph_pred.shape[0], offset=1)
    triu_mask = torch.squeeze(to_dense_adj(triu_indices)).bool()
    graph_pred_triu= graph_pred[triu_mask]
    return graph_pred_triu
def loss_fcn(triu_logits, edge_index, mu, logvar, batch_index, kl_beta=0.01):
#     print(edge_index.size())
    batch_targets = torch.squeeze(to_dense_adj(edge_index))

    batch_recon_loss = []
    batch_node_counter = 0

    # Loop over graphs in this batch
    for graph_id in torch.unique(batch_index):
        # Get upper triangular targets for this graph from the whole batch
        graph_targets_triu = slice_graph_targets(graph_id, 
                                                batch_targets, 
                                                batch_index)
#         print(graph_targets_triu)

        # Get upper triangular predictions for this graph from the whole batch
        graph_predictions_triu = slice_graph_pred(graph_id, triu_logits, batch_index)

        # Update counter to the index of the next graph
#         batch_node_counter = batch_node_counter + graph_targets_triu.shape[0]

        # Calculate edge-weighted binary cross entropy
        weight = graph_targets_triu.shape[0]/sum(graph_targets_triu)
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=weight).to(device)
        graph_recon_loss = bce(graph_predictions_triu.view(-1), graph_targets_triu.view(-1))
        batch_recon_loss.append(graph_recon_loss)   

    # Take average of all losses
    num_graphs = torch.unique(batch_index).shape[0]
    batch_recon_loss = sum(batch_recon_loss) / num_graphs

    total_loss= batch_recon_loss 

    return total_loss


# In[10]:


def run_one_epoch(train_loader,epoch):
    model.train()
    total_loses=[]
    total_graph=[]
    reconstructed_graph = 0
    KL_loses=[]
    MSE_loses=[]
    for _, data in enumerate(train_loader):
      optimizer.zero_grad()
      data= data.to(device)
      mu, logvar, embeddings, edges_pred= model(data.x, data.edge_index,data.batch)
      edge_target = data.edge_index
      edge_loss= loss_fcn(edges_pred, data.edge_index, mu, logvar, data.batch)  
#       loss_nll =nll_gaussian(node_features, data.x, 0.001)
#       node_loss_mse = F.mse_loss(node_features, data.y, reduction='mean') 
    #   print(loss_nll, )
      kl_divergence= model.kl_loss(mu,logvar)
      kl_beta= 1  
      loss = edge_loss +kl_beta* kl_divergence    
      loss.backward()
      total_loses.append(loss.detach().cpu().numpy())
#       MSE_loses.append(node_loss_mse.detach().cpu().numpy())
      KL_loses.append(kl_divergence.detach().cpu().numpy())
      optimizer.step()
    if(epoch % 20 == 0):        
        print("epoch:", epoch+1, "loss:" , np.array(total_loses).mean(),"KL loss:" , np.array(KL_loses).mean())
    # print(f"{type} epoch {epoch+1} accuracy: ", np.array(all_accs).mean())
    # print(f"Reconstructed {reconstructed_graph} out of {total_graph} graphs.")

    # print("\n****###**###**###****\n")
    return KL_loses, total_loses

# Run training
# train(train_loader)
KL_loses_epoch=[]
# NLL_loses_epoch=[]
total_loses_epoch=[]
n_epoch=100
for epoch in range(n_epoch): 
    # model.train()
  KL_loses,total_loses=run_one_epoch(train_loader,epoch)
  KL_loses_epoch.append(KL_loses)
  total_loses_epoch.append(total_loses)


# In[103]:


from typing import Any, Optional
from math import sqrt

def visualize_graph_via_networkx(
    edge_index: Tensor, edge_weight: Optional[Tensor] = None):
    import matplotlib.pyplot as plt
    import networkx as nx

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1))

    g = nx.Graph()
    node_size = 800

    for node in edge_index.view(-1).unique().tolist():
        g.add_node(node)

    for (src, dst), w in zip(edge_index.t().tolist(), edge_weight.tolist()):
        g.add_edge(src, dst, alpha=w)

    ax = plt.gca()
    pos = nx.spring_layout(g)
    for src, dst, data in g.edges(data=True):
        ax.annotate(
            '',
            xy=pos[src],
            xytext=pos[dst],
            arrowprops=dict(
                arrowstyle="-",
                alpha=data['alpha'],
                shrinkA=sqrt(node_size) / 2.0,
                shrinkB=sqrt(node_size) / 2.0,
                connectionstyle="arc3,rad=0",
            ),
        )

    nodes = nx.draw_networkx_nodes(g, pos, node_size=node_size,
                                   node_color='slateblue', margins=0.1)
    colors = range(20)
    options = {
#         "node_color": "#A0CBE2",
        "edge_color": colors,
        "width": 4,
        "edge_cmap": plt.cm.Blues,
        "with_labels": False,
    }
    #     edges= nx.draw_networkx_edges(g, pos, edgelist=None, width=1.0, edge_color='blue')
#     nodes.set_edgecolor('red')
#     nodes.set_sizes
    nx.draw_networkx_labels(g, pos, font_size=10)

#     nx.draw_networkx_edges(g, pos,edgelist = g.edges(), width=1.0, edge_color='k', style='solid')
    plt.show()

    plt.close()


# In[118]:


model.eval()
data= next(iter(test_loader))
data = data.to(device)
mu, logvar, embeddings,edges_adj = model(data.x, data.edge_index,data.batch)
# print(node_features.size())
# print(edges_adj.size())

graph_id=55
edges_adj =edges_adj[10*graph_id:10*(graph_id+1)]
print(edges_adj.size())
edges_adj = edges_adj.detach().cpu().numpy()
edges_adj=1*edges_adj>0.40
edges_adj= edges_adj.astype('uint8')
# print(edges_adj,'ea')
edge_adjacency_tensor = torch.tensor(edges_adj)
edge_list = dense_to_sparse(edge_adjacency_tensor)
# print(edge_list[0])
visualize_graph_via_networkx(edge_index=edge_list[0])


# In[ ]:





# In[54]:


mu, logvar, embeddings,edges_adj = model(data.x, data.edge_index,data.batch)
print(data.edge_index.size())


# In[55]:


model.eval()
for data in enumerate(test_loader):
    data = data.to(device)
    edges_gt=data.edge_index
    mu, logvar, embeddings,edges_adj = model(data.x, data.edge_index,data.batch)
    
    try:
        edges_pred = torch.cat((edges_pred, edges_adj), dim=0)
    except:
        edges_pred = edges_adj
    try:
        g_truth = torch.cat((g_truth, target), dim=0)
    except:
        g_truth = target


# In[15]:


from torch_geometric.visualization import visualize_graph
visualize_graph(edge_index=edge_list[0], backend='networkx')


# In[ ]:





# In[64]:


edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                           [1, 0, 2, 1, 3, 2]])
adj = to_torch_coo_tensor(edge_index)
print(adj)
new_adj=to_edge_index(adj)
print(new_adj)


# In[19]:


from torch_geometric.utils import to_networkx, to_edge_index
import networkx as nx


# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
graph_z = dense_to_sparse(graph_z)[0]
print(graph_z)
data =Data(x=node_features, edge_index=graph_z)
# data
G = to_networkx(data, to_undirected=True)
# nx.draw(g)
G
# G = to_networkx(graph_z, to_undirected=True)
plt.figure(figsize=(12,12))
plt.axis('off')
nx.draw_networkx(G,
                pos=nx.spring_layout(G, seed=0),
                with_labels=True,
                node_size=800,
#                 cmap="hsv",
                vmin=-2,
                vmax=3,
                width=0.8,
                edge_color="grey",
                font_size=14
                )
plt.show()


# In[65]:


import networkx as nx

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data =Data(x=x, edge_index=edge_index)
g =to_networkx(data, to_undirected=True)
nx.draw_networkx(g)


# In[24]:


test_predictions = np.zeros([300,10,4])
test_actual = np.zeros([300,10,4])

model.eval()
for batch_id, data in enumerate(test_loader):
  mu, logvar, embeddings, node_features, edges_adj = model(data.x, data.edge_index,data.batch)
  pred=node_features.detach().cpu().numpy().reshape(-1,10,4)
  test_predictions=np.concatenate((test_predictions, pred.reshape(-1,10,4)), axis=0)
  test_actual=np.concatenate((test_actual, data.x.reshape(-1,10,4)), axis=0)

test_predictions= test_predictions[200:,:,:]
test_predictions[:,:,1:2]= urnnormalize(test_predictions[:,:,1:2], loc_min, loc_max)
test_predictions[:,:,2:4]= urnnormalize(test_predictions[:,:,2:4], vel_min, vel_max)
test_actual[:,:,1:2]= urnnormalize(test_actual[:,:,1:2], loc_min, loc_max)


# In[26]:


pl=3
for i in range(pl):
    plt.figure()
    plt.plot(test_predictions[:,i,0],test_predictions[:,i,1], 'ro')
    # plt.plot(test_actual[:,i,0],test_actual[:,i,1], 'bo')


# In[46]:


import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GAE, VGAE, InnerProductDecoder
from torch_geometric.utils import train_test_split_edges

from tqdm import tqdm


class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True)
        # self.mean = torch.nn.Linear(hidden_size, latent_size)
        # self.log_std = torch.nn.Linear(hidden_size, latent_size)
        self.conv_mean = GCNConv(hidden_size, latent_size)
        self.conv_logvar = GCNConv(hidden_size, latent_size)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = x.unsqueeze(0)
        x, _ = self.lstm(x)
        x = x.squeeze(0)
        z_mean = self.conv_mean(x)
        z_log_std = self.conv_logvar(x)
        return z_mean, z_log_std


class Decoder(torch.nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.lstm = torch.nn.LSTM(latent_size, hidden_size, batch_first=True)
        self.gc1 = GCNConv(hidden_size, hidden_size)
        self.gc2 = GCNConv(hidden_size, hidden_size)
        self.linear1 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, z, edge_index):
        z = z.unsqueeze(1)
        z, _ = self.lstm(z)
        z = z.squeeze(1)
        x = F.relu(self.gc1(z, edge_index))
        x = self.gc2(x, edge_index)
        x = F.relu(self.linear1(x))
        return x


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, z_mean, z_log_std):
        eps = torch.randn_like(z_mean)
        return eps.mul(torch.exp(z_log_std)).add_(z_mean)

    def forward(self, data):
        z_mean, z_log_std = self.encoder(data)
        z = self.reparameterize(z_mean, z_log_std)
        return self.decoder(z, data.edge_index), z_mean, z_log_std

    def loss(self, recon_x, x, z_mean, z_log_std):
        BCE = F.mse_loss(recon_x, x, reduction='mean')
        KLD = -0.5 * torch.mean(1 + z_log_std - z_mean.pow(2) - z_log_std.exp())
        loss= BCE+KLD
        return loss
encoder =Encoder(input_size=4, hidden_size=64, latent_size=4)
decoder =Decoder(latent_size=4, hidden_size=64, output_size=4)
model = VAE(encoder, decoder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
model
# @torch.no_grad()
# def test(model, loader, device):
#     model.eval()

#     total_loss = 0
#     for data in tqdm(loader, desc='Testing', leave


# In[44]:


class VAE(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, lstm_hidden_size):
        super(VAE, self).__init__()
        self.num_layers = num_layers
        self.lstm_hidden_size = lstm_hidden_size
        
        self.conv1 = GCNConv(4, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        self.encoder_lstm = torch.nn.LSTM(input_size=hidden_channels, hidden_size=lstm_hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder_lstm = torch.nn.LSTM(input_size=4, hidden_size=lstm_hidden_size, num_layers=num_layers, batch_first=True)
        
        self.conv_mu = GCNConv(lstm_hidden_size, 4)
        self.conv_logstd = GCNConv(lstm_hidden_size, 4)

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # reshape x to be a sequence of feature vectors for the LSTM
        batch_size = x.size(0) // self.num_layers
        x = x.view(batch_size, self.num_layers, -1)
        
        h_0 = torch.zeros(self.num_layers, batch_size, self.lstm_hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.lstm_hidden_size).to(x.device)
        _, (h_n, _) = self.encoder_lstm(x, (h_0, c_0))
        
        # concatenate the final hidden states of the LSTM layers
        z = h_n[-1]
        
        return self.conv_mu(z), self.conv_logstd(z)

    def reparameterize(self, mu, logstd):
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def decode(self, z, edge_index, seq_len):
        # repeat the latent variable for each time step
        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        
        h_0 = torch.zeros(self.num_layers, z.size(0), self.lstm_hidden_size).to(z.device)
        c_0 = torch.zeros(self.num_layers, z.size(0), self.lstm_hidden_size).to(z.device)
        z, _ = self.decoder_lstm(z, (h_0, c_0))
        z = z.view(-1, self.lstm_hidden_size)
        
        x = F.relu(self.conv1(z, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.conv_mu(x), self.conv_logstd(x)

    def forward(self, x, edge_index, seq_len):
        mu, logstd = self.encode(x, edge_index)
        z = self.reparameterize(mu, logstd)
        return self.decode(z, edge_index, seq_len), mu, logstd

model = VAE(hidden_channels=64, num_layers=2, lstm_hidden_size=128)


# In[49]:


def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    for batch,data in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_data, z_mean, z_log_std = model(data)
        loss = model.loss(recon_data, data.x, z_mean, z_log_std)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)
train(model, optimizer, train_loader)


# In[33]:


len(train_loader.dataset)


# In[ ]:


test_predictions = np.zeros([1900,10,4])
model.eval()
for batch_id, data in enumerate(test_loader):
  mu, logvar, embeddings, node_features, edges_adj = model(data.x, data.edge_index,data.batch)
  # edges_adj=model.decode_edges(z)
  edge_threshold=0.5
  edge_preds = (edges_adj > edge_threshold).float()
  # print(edge_preds)
  # print(recon_x)

