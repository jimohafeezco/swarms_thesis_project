#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sklearn.metrics as metrics

# Visualization libraries
import matplotlib.pyplot as plt
import networkx as nx
# from torch_geometric.datasets import KarateClub


import os.path as osp
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_torch_coo_tensor
# from torch_geometric.utils import to_edge_index
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from torch.nn import Linear
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn import Set2Set
from torch_geometric.nn import BatchNorm


# In[2]:



from torch import Tensor
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric_temporal.nn.recurrent import GConvGRU, DyGrEncoder, EvolveGCNH
from torch_geometric_temporal.nn.recurrent import DCRNN

from torch_geometric.nn import global_mean_pool, global_add_pool

from torch_geometric.nn import GAE, VGAE, GCNConv
import torch.nn.functional as F
import numpy as np
import pandas as pd
# import npzviewer

from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader


# In[3]:



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

features, edges,loc_max, loc_min, vel_max, vel_min = load_motion_data(batch_size=32)
features_tensor= torch.FloatTensor(features)
edges_tensor = torch.FloatTensor(edges)

train_tensor = TensorDataset(features_tensor, edges_tensor)
print(np.shape(features),'features')
print(np.shape(edges),'edges')

data_list = []
N=19000
for i in range(np.shape(edges)[0]-1):
  edge_list = dense_to_sparse(edges_tensor[i])
  # print(edge_list)
  data=Data(x=features_tensor[i], edge_index=edge_list[0], y = features_tensor[i+1])
  data_list.append(data)
# loader = DataLoader(data_list)
dataset= data_list
print(len(dataset))
# data_agents= features
# plt.figure()
# for i in range(0,3):
#     plt.scatter(data_agents[0:1000, i,0],data_agents[0:1000,i,1])
# #     plt.plot(data_agents[0:1000,i,0],data_agents[0:1000,i,1],'r--')
# plt.show()
# plt.figure()
# for i in range(0,3):
#     plt.scatter(data_agents[-1000:, i,0],data_agents[-1000:,i,1])
# #     plt.plot(data_agents[-1000:,i,0],data_agents[-1000:,i,1],'r--')
# plt.show()


# In[4]:


from torch_geometric.loader import DataLoader

# Create training, validation, and test sets
train_dataset = dataset[:int(len(dataset)*0.8)]
val_dataset   = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]
test_dataset  = dataset[int(len(dataset)*0.9):]

print(f'Training set   = {len(train_dataset)} graphs')
print(f'Validation set = {len(val_dataset)} graphs')
print(f'Test set       = {len(test_dataset)} graphs')
batch_size= 64

# Create mini-batches
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size,shuffle=0)
test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=0)
test2_loader = DataLoader(test_dataset,batch_size=1, shuffle=0)


# In[5]:


data= next(iter(test_loader))
data=data.x.reshape(-1,10,4)
print(data.shape)
c=['ro','bo','go']
# fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10, 5))
for i in range(0,3):
#     plt.figure()
    plt.plot(data[:,i,0],data[:,i,1],c[i])
    plt.plot(data[:,i,0],data[:,i,1],c[i])
    plt.plot(data[:,i,0],data[:,i,1],c[i])


# ## **Using DYGREncoder**

# In[6]:


class RecurrentDyGrEncoder(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentDyGrEncoder, self).__init__()
        self.recurrent = DyGrEncoder(conv_out_channels=32, conv_num_layers=3, conv_aggr="mean", lstm_out_channels=32, lstm_num_layers=1)
        self.linear = torch.nn.Linear(32, 4)    
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.0)
                    
    def forward(self, x, edge_index, edge_weight, h_0, c_0):
        h, h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h_0, c_0)
        h = F.relu(h)
        h = self.linear(h)
        adj = F.sigmoid( torch.matmul(h, h.t()))
        return h,adj
        
model = RecurrentDyGrEncoder(node_features = 4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model= model.to(device)
model


# In[7]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model= model.to(device)

from tqdm import tqdm
import time
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
# scheduler = ExponentialLR(optimizer, gamma=0.9)

model.train()
n_epochs=200
for epoch in range(n_epochs):
    weight=None
    h, c = None, None
    t = time.time()
    cost = 0
    for batch_id, data in enumerate(train_loader):
        data= data.to(device)
#         print(data.x.size())
        y_hat,_= model(data.x, data.edge_index, weight, h,c)
#         print(adj.size())
#         print(adj)

        cost = cost + F.mse_loss(y_hat,data.y)

    cost = cost / (batch_id+1)
    cost.backward()
    optimizer.step()
#     scheduler.step()
    optimizer.zero_grad()
    if(epoch % 20 == 0):
        print('Epoch: {:04d}'.format(epoch+1),
              'mse_train: {:.10f}'.format(cost),
              'time: {:.4f}s'.format(time.time()- t)) 
#     return cost


# In[9]:


weight=None
pred=0
g_truth=0
del pred
del g_truth
h, c = None, None
model.eval()
cost = 0
for time, snapshot in enumerate(test_loader):
    snapshot= snapshot.to(device)
    y_hat,_= model(snapshot.x, snapshot.edge_index, weight, h,c)
#     cost = cost + nn.MSELoss(snapshot.y, y_hat)
#     print(snapshot.x.size())
    cost = cost+  torch.mean((y_hat-snapshot.y)**2)
    cost = cost / (time+1)
    try:
        pred = torch.cat((pred, y_hat), dim=0)
    except:
        pred = y_hat
    try:
        g_truth = torch.cat((g_truth, snapshot.y), dim=0)
    except:
        g_truth = snapshot.y
cost = cost.item()
print("MSE: {:6f}".format(cost))
print(np.shape(pred))
predi=pred.squeeze().detach().cpu().numpy().reshape(-1,10,4)
g_truth =g_truth.squeeze().detach().cpu().numpy().reshape(-1,10,4)
print(np.shape(predi))


# In[11]:


snapshot = next(iter(test2_loader))
# print(snapshot.edge_index.size())
fixed_adj= snapshot.edge_index
fixed_x= snapshot.x+0.2
print(fixed_x)

import pandas as pd    
# del pred
model.eval()
weight=None
h, c = None, None
for i in range(200):
    y_hat, adj= model(fixed_x.to(device), fixed_adj.to(device), weight, h,c)
    fixed_x = y_hat
#      print(fixed_x)
    try:
        pred = torch.cat((pred, fixed_x), dim=0)
    except:
        pred = fixed_x
predi=pred.squeeze().detach().cpu().numpy().reshape(-1,10,4)

print(np.shape(predi))

def urnnormalize(data, data_max, data_min):
	return (data + 1) * (data_max - data_min) / 2. + data_min

pred_position = predi[:,:,0:2]

pred_positionU = urnnormalize(pred_position, loc_max, loc_min)
plt.plot(pred_positionU[:,6,0],pred_positionU[:,6,1])
plt.xlabel("x position")
plt.ylabel("y position")


# In[10]:


colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w','teal','hotpink']
fig, ax = plt.subplots()
for i in range(10):
    x=pred_positionU[:,i,0]
    y =pred_positionU[:,i,1]   

    plt.plot(pred_positionU[:,i,0],pred_positionU[:,i,1],c=colors[i])
    plt.xlabel("x position")
    plt.ylabel("y position")
#     plt.savefig('images/'+'infinitemotion_boids1',dpi=300)


# In[ ]:



import itertools
import itertools
from collections import deque
from IPython import display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
agentPositions= pred_positionU
dataX= agentPositions[:,:,0]
dataY= agentPositions[:,:,1]
xmin = dataX.min()
ymin = dataY.min()
xmax = dataX.max()
ymax = dataY.max()
# data1 = np.load(data/edges_test_springs5.npy)
def animate_func(i):
    ax.clear()  # Clears the figure to update the line, point,   

    ax.scatter(dataX[i,:], dataY[i,:], c='red', marker='o',label='prediction')
    ax.set_xlim(xmin-0.5, xmax+0.5)
    ax.set_ylim(ymin-0.5, ymax+0.5)
    ax.legend()
# numDataPoints=49
fig, ax = plt.subplots()
line_ani = animation.FuncAnimation(fig, animate_func, interval=40,   
                                   frames=np.shape(dataX)[0])
# plt.show()

line_ani.save('images/'+'pred_infin.gif', writer='pillow')


# In[ ]:


snapshot = next(iter(test2_loader))
# print(snapshot.edge_index.size())
fixed_adj= snapshot.edge_index
fixed_x= snapshot.x+0.6
print(fixed_x)

import pandas as pd    
del pred
model.eval()
weight=None
h, c = None, None
for i in range(200):
    y_hat, adj= model(fixed_x.to(device), fixed_adj.to(device), weight, h,c)
    fixed_x = y_hat
#      print(fixed_x)
    try:
        pred = torch.cat((pred, fixed_x), dim=0)
    except:
        pred = fixed_x
predi=pred.squeeze().detach().cpu().numpy().reshape(-1,10,4)

print(np.shape(predi))

def urnnormalize(data, data_max, data_min):
	return (data + 1) * (data_max - data_min) / 2. + data_min

pred_position = predi[:,:,0:2]

pred_positionU = urnnormalize(pred_position, loc_max, loc_min)


colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w','teal','hotpink']
fig, ax = plt.subplots()
for i in range(10):
    x=pred_positionU[:,i,0]
    y =pred_positionU[:,i,1]   

    plt.plot(pred_positionU[:,i,0],pred_positionU[:,i,1],c=colors[i])
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.savefig('images/'+'infinitemotion_boids2',dpi=300)


# In[ ]:



import itertools
import itertools
from collections import deque
from IPython import display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
agentPositions= pred_positionU
dataX= agentPositions[:,:,0]
dataY= agentPositions[:,:,1]
xmin = dataX.min()
ymin = dataY.min()
xmax = dataX.max()
ymax = dataY.max()
# data1 = np.load(data/edges_test_springs5.npy)
def animate_func(i):
    ax.clear()  # Clears the figure to update the line, point,   

    ax.scatter(dataX[i,:], dataY[i,:], c='red', marker='o',label='prediction')
    ax.set_xlim(xmin-0.5, xmax+0.5)
    ax.set_ylim(ymin-0.5, ymax+0.5)
    ax.legend()
# numDataPoints=49
fig, ax = plt.subplots()
line_ani = animation.FuncAnimation(fig, animate_func, interval=40,   
                                   frames=np.shape(dataX)[0])
# plt.show()

line_ani.save('images/'+'pred_infin2.gif', writer='pillow')


# In[ ]:


colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w','teal','hotpink']
fig, ax = plt.subplots()
for i in range(10):
    r = np.round(np.random.rand(),1)
    g = np.round(np.random.rand(),1)
    b = np.round(np.random.rand(),1)
    a = np.round(np.clip(np.random.rand(), 0, 1), 1)
    x=pred_positionU[:,i,0]
    y =pred_positionU[:,i,1]   
    x=pd.DataFrame(x)
    y=pd.DataFrame(y)

    x0 = x.iloc[range(len(x)-1)].values
    x1 = x.iloc[range(1,len(x))].values
    y0 = y.iloc[range(len(y)-1)].values
    y1 = y.iloc[range(1,len(y))].values
    xpos = (x0+x1)/2
    ypos = (y0+y1)/2
    xdir = x1-x0
    ydir = y1-y0
#     fig, ax = plt.subplots()
#     # ax.scatter(x,y)
    ax.plot(x,y,c='white')
    # plot arrow on each line:
    for X,Y,dX,dY in zip(xpos, ypos, xdir, ydir):
        ax.annotate("", xytext=(X,Y),xy=(X+0.1*dX,Y+0.1*dY), 
        arrowprops=dict(arrowstyle="->", color=colors[i]), size = 10)
#     plt.plot(pred_positionU[:,i,0],pred_positionU[:,i,1],c=colors[i])
#     plt.xlabel("x position")
#     plt.ylabel("y position")
#     plt.savefig('images/'+'infinitemotion_boids',dpi=300)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w','teal','hotpink']
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7','C8','C9']

fig, ax = plt.subplots()
for i in range(10):
    r = np.round(np.random.rand(),1)
    g = np.round(np.random.rand(),1)
    b = np.round(np.random.rand(),1)
    a = np.round(np.clip(np.random.rand(), 0, 1), 1)
    x=pred_positionU[:,i,0]
    y =pred_positionU[:,i,1]   
#     x=pd.DataFrame(x)
#     y=pd.DataFrame(y)
    u = np.diff(x)
    v = np.diff(y)
    pos_x = x[:-1] + u/2
    pos_y = y[:-1] + v/2
    norm = np.sqrt(u**2+v**2) 

#     fig, ax = plt.subplots()
    # ax.plot(x,y, marker="o")
    ax.quiver(pos_x, pos_y, u/norm, v/norm,color=colors[i],angles="xy", zorder=5, units='width')
plt.show()


# In[11]:


pred=0
del pred
model.eval()
weight=None
h, c = None, None
for batch_id, data in enumerate(test_loader):
    data= data.to(device)
    y_hat,_= model(data.x, data.edge_index, weight, h,c)
#     print(y_hat.size())
    try:
        pred = torch.cat((pred, y_hat), dim=0)
    except:
        pred = y_hat
        
y_pred=pred.squeeze().detach().cpu().numpy().reshape(-1,10,4)
print(y_pred.shape)
# y_pred =[model(data.x.to(device), data.edge_index.to(device), weight, h,c).squeeze().detach().cpu().numpy().reshape(-1,10,4) for data in test_loader]
# y_pred=np.concatenate(y_pred, axis=0)

target_test =[data.x.squeeze().detach().cpu().numpy().reshape(-1,10,4) for data in test_loader]
target_test=np.concatenate(target_test, axis=0)

# print(y_pred.size())
pred_position = y_pred[:,:,0:2]
pred_velocity = y_pred[:,:,2:4]
gt_position = target_test[:,:,0:2]
gt_velocity = target_test[:,:,2:4]
# print(features_test.shape)
def urnnormalize(data, data_max, data_min):
	return (data + 1) * (data_max - data_min) / 2. + data_min

pred_positionU = urnnormalize(pred_position, loc_max, loc_min)
pred_velocityU = urnnormalize(pred_velocity, vel_max, vel_min)
# target

gt_positionU = urnnormalize(gt_position, loc_max, loc_min)
gt_velocityU = urnnormalize(gt_velocity, vel_max, vel_min)

c=['ro','bo','go']
# fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10, 5))


# In[19]:


def metric_amd(predict):
    T,N,D=np.shape(predict)
    amd = 0
    adv= 0
    for j in range(2):
        for i in range(10):
            predi_neigh= np.delete(predict, i, 1)
#             print(np.shape(predi_neigh))
            dist_diff=np.abs(predict[:,i,j].reshape(1900,1)-predi_neigh[:,:,j])
            amd=amd+(np.min(dist_diff, axis=1))
#             print(amd)
    return amd/(N)

def metric_avd(predict):
    N,T,D=np.shape(predict)
    amd = 0
    adv= 0
    for j in range(0,2):
        for i in range(10):
            predi_neigh= np.delete(predict, i, 0)
            dist_diff=np.abs(predict[i,:,j]-predi_neigh[:,:,j])
            amd=amd+(np.sum(dist_diff, axis=1))
#             print(amd)
    return 2*amd/(N*(N-1))

step=10
# mse_loss, mae_Loss, mape_loss, predi, g_truth=test_model(test_loader, step)

amd_total_predi=metric_amd(pred_positionU)
amd_total_gtruth=metric_amd(gt_positionU)
avd_total_predi=metric_avd(pred_velocityU)
avd_total_gtruth=metric_avd(gt_velocityU)


amd_total = amd_total_predi.mean()
avd_total = avd_total_predi.mean()
avd_total_gt = avd_total_gtruth.mean()
amd_total_gt = amd_total_predi.mean()
print(amd_total,"amd total")
print(avd_total,"avd total")
print(amd_total_gt,"amd total")
print(avd_total_gt,"avd total")
# print(np.shape(avd_total))
plt.plot(amd_total_predi, label="Prediction",c='orangered')
plt.plot(amd_total_gtruth,label="Ground truth",c='dodgerblue')
plt.xlabel("Time")
plt.ylabel("AMD")
plt.legend()
plt.figure()
plt.plot(avd_total_predi, label="Prediction",c='orangered')
plt.plot(avd_total_gtruth,label="Ground truth",c='dodgerblue')
plt.xlabel("Time")
plt.ylabel("AVD")
plt.legend()


# In[ ]:


# for i in range(0,9):
#     plt.plot(gt_positionU[:,i,0],gt_positionU[:,i,1],'r--')
#     plt.plot(pred_positionU[:,i,0],pred_positionU[:,i,1],'g--')
#     plt.figure()


# ## USING DCRNN

# In[16]:


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 64, 3)
        self.linear = torch.nn.Linear(64, 4)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h
        
model = RecurrentGCN(node_features = 4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model= model.to(device)
model


# In[17]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model= model.to(device)

from tqdm import tqdm
import time
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = ExponentialLR(optimizer, gamma=0.9)

model.train()
n_epochs=200
for epoch in range(n_epochs):
    weight=None
    h, c = None, None
    t = time.time()
    cost = 0
    for batch_id, data in enumerate(train_loader):
        data= data.to(device)
        y_hat = model(data.x, data.edge_index, weight)
#         print(y_hat.size())
        cost = cost + F.mse_loss(y_hat,data.y)

    cost = cost / (batch_id+1)
    cost.backward()
    optimizer.step()
#     scheduler.step()
    optimizer.zero_grad()
    if(epoch % 20 == 0):
        print('Epoch: {:04d}'.format(epoch+1),
              'mse_train: {:.10f}'.format(cost),
              'time: {:.4f}s'.format(time.time()- t)) 
    # return cost


# In[18]:


weight=None
h, c = None, None
model.eval()
cost = 0
for time, snapshot in enumerate(test_loader):
    snapshot= snapshot.to(device)
    y_hat= model(snapshot.x, snapshot.edge_index,weight)
    cost = cost + torch.mean((y_hat-snapshot.y)**2)
cost = cost / (time+1)
cost = cost.item()
print("MSE: {:.4f}".format(cost))


# In[19]:


weight=None
pred=0
g_truth=0
del pred
del g_truth
h, c = None, None
model.eval()
cost = 0
for time, snapshot in enumerate(test_loader):
    snapshot= snapshot.to(device)
    y_hat= model(snapshot.x, snapshot.edge_index,weight)
#     cost = cost + nn.MSELoss(snapshot.y, y_hat)
#     print(snapshot.x.size())
    cost = cost+  torch.mean((y_hat-snapshot.y)**2)
    cost = cost / (time+1)
    try:
        pred = torch.cat((pred, y_hat), dim=0)
    except:
        pred = y_hat
    try:
        g_truth = torch.cat((g_truth, snapshot.y), dim=0)
    except:
        g_truth = snapshot.y
cost = cost.item()
print("MSE: {:6f}".format(cost))
print(np.shape(pred))
predi=pred.squeeze().detach().cpu().numpy().reshape(-1,10,4)
g_truth =g_truth.squeeze().detach().cpu().numpy().reshape(-1,10,4)
print(np.shape(predi))

def normalize(data, data_max, data_min):
    return (data - data_min) * 2 / (data_max - data_min) - 1

# loc_max, loc_min, vel_max, vel_min
def urnnormalize(data, data_max, data_min):
    return (data + 1) * (data_max - data_min) / 2. + data_min

position_unnormalize = urnnormalize(predi[:,:,0:2], loc_max, loc_min)
velocity_unnormalize = urnnormalize(predi[:,:,2:4], vel_max, vel_min)
# target

gt_position_unnormalize = urnnormalize(g_truth[:,:,0:2], loc_max, loc_min)
gt_velocity_unnormalize = urnnormalize(g_truth[:,:,2:4], vel_max, vel_min)

predicted_features = np.concatenate([position_unnormalize, velocity_unnormalize],axis=2)
#     print(predicted_features.shape)
target_features =np.concatenate([gt_position_unnormalize, gt_velocity_unnormalize ],axis=2)
#


# In[20]:


def metric_amd(predict):
    T,N,D=np.shape(predict)
    amd = 0
    adv= 0
    for j in range(2):
        for i in range(10):
            predi_neigh= np.delete(predict, i, 1)
#             print(np.shape(predi_neigh))
            dist_diff=np.abs(predict[:,i,j].reshape(-1,1)-predi_neigh[:,:,j])
            amd=amd+(np.min(dist_diff, axis=1))
#             print(amd)
    return amd/(N)

def metric_avd(predict):
    T,N,D=np.shape(predict)
    amd = 0
    adv= 0
    for j in range(2,4):
        for i in range(10):
            predi_neigh= np.delete(predict, i, 1)
            dist_diff=np.abs(predict[:,i,j].reshape(-1,1)-predi_neigh[:,:,j])
            amd=amd+(np.min(dist_diff, axis=1))
#             print(amd)
    return amd/(N)
step=10
# mse_loss, mae_Loss, mape_loss, predi, g_truth=test_model(test_loader, step)

amd_total_predi=metric_amd(predicted_features)
amd_total_gtruth=metric_amd(target_features)
avd_total_predi=metric_avd(predicted_features)
avd_total_gtruth=metric_avd(target_features)


amd_total = amd_total_predi.mean()
avd_total = avd_total_predi.mean()
avd_total_gt = avd_total_gtruth.mean()
amd_total_gt = amd_total_predi.mean()
print(amd_total,"amd total")
print(avd_total,"avd total")
print(amd_total_gt,"amd total")
print(avd_total_gt,"avd total")
# print(np.shape(avd_total))
plt.plot(amd_total_predi, label="Prediction",c='orangered')
plt.plot(amd_total_gtruth,label="Ground truth",c='dodgerblue')
plt.xlabel("Time")
plt.ylabel("AMD")
plt.legend()
plt.figure()
plt.plot(avd_total_predi, label="Prediction",c='orangered')
plt.plot(avd_total_gtruth,label="Ground truth",c='dodgerblue')
plt.xlabel("Time")
plt.ylabel("AVD")
plt.legend()


# In[21]:


model.eval()
data= next(iter(test_loader))
data= data.to(device)
y_hat= model(data.x, data.edge_index, weight)
data=data.y.reshape(-1,10,4)
features_test = y_hat.detach().cpu().numpy()
target_test = data.detach().cpu().numpy()
features_test=features_test.reshape(-1,10,4)

print(y_hat.size())
pred_position = features_test[:,:,0:2]
pred_velocity = features_test[:,:,2:4]
gt_position = target_test[:,:,0:2]
gt_velocity = target_test[:,:,2:4]
print(features_test.shape)

def urnnormalize(data, data_max, data_min):
	return (data + 1) * (data_max - data_min) / 2. + data_min

pred_positionU = urnnormalize(pred_position, loc_max, loc_min)
pred_velocityU = urnnormalize(pred_velocity, vel_max, vel_min)
# target

gt_positionU = urnnormalize(gt_position, loc_max, loc_min)
gt_velocityU = urnnormalize(gt_velocity, vel_max, vel_min)

c=['ro','bo','go']
# fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10, 5))
for i in range(0,9):
#     plt.plot(gt_positionU[:,i,0],gt_positionU[:,i,1],'ro')
    plt.plot(pred_positionU[:,i,0],pred_positionU[:,i,1],'go')
    plt.figure()


# In[22]:


data= next(iter(test_loader))
data= data.to(device)
y_hat ,_,_= model(data.x, data.edge_index, weight, h,c)
data=data.y.reshape(-1,10,4)
features_test = y_hat.detach().cpu().numpy()
target_test = data.detach().cpu().numpy()
features_test=features_test.reshape(-1,10,4)

print(y_hat.size())
pred_position = features_test[:,:,0:2]
pred_velocity = features_test[:,:,2:4]
gt_position = target_test[:,:,0:2]
gt_velocity = target_test[:,:,2:4]
print(features_test.shape)
def urnnormalize(data, data_max, data_min):
	return (data + 1) * (data_max - data_min) / 2. + data_min

pred_positionU = urnnormalize(pred_position, loc_max, loc_min)
pred_velocityU = urnnormalize(pred_velocity, vel_max, vel_min)
# target

gt_positionU = urnnormalize(gt_position, loc_max, loc_min)
gt_velocityU = urnnormalize(gt_velocity, vel_max, vel_min)

c=['ro','bo','go']
# fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10, 5))
for i in range(0,9):
    plt.plot(gt_positionU[:,i,0],gt_positionU[:,i,1],'ro')
    plt.plot(pred_positionU[:,i,0],pred_positionU[:,i,1],'go')
    plt.figure()


# In[ ]:




