#!/usr/bin/env python
# coding: utf-8

# Import Libraries

# In[1]:


import numpy as np
import torch


# Create 1D 2D 3D Tensors

# In[2]:


np_1d = np.array([1, 2, 3, 4])
np_2d = np.array([[1, 2, 3], [4, 5, 6]])
np_3d = np.arange(24).reshape(2, 3, 4)
torch_1d = torch.tensor([1, 2, 3, 4])
torch_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
torch_3d = torch.arange(24).reshape(2, 3, 4)
np_1d, np_2d, np_3d.shape, torch_1d, torch_2d, torch_3d.shape


# Element Wise Operations

# In[3]:


np_a = np.array([1, 2, 3, 4])
np_b = np.array([10, 20, 30, 40])
np_a + np_b, np_a * np_b
torch_a = torch.tensor([1, 2, 3, 4])
torch_b = torch.tensor([10, 20, 30, 40])
torch_a + torch_b, torch_a * torch_b


# Indexing and Slicing

# In[4]:


np_x = np.arange(20).reshape(4, 5)
torch_x = torch.arange(20).reshape(4, 5)
np_x[1, 2], torch_x[1, 2]
np_x[:, 1:4], torch_x[:, 1:4]
np_mask = np_x % 2 == 0
torch_mask = torch_x % 2 == 0
np_x[np_mask], torch_x[torch_mask]
np_x[1:3, 2:5], torch_x[1:3, 2:5]


# View Reshape Unsqueeze Squeeze

# In[5]:


t = torch.arange(12)
t_view = t.view(3, 4)
t_reshape = t.reshape(2, 2, 3)
t_unsq0 = t.unsqueeze(0)
t_unsq1 = t.unsqueeze(1)
t_sq = t_unsq0.squeeze()
n = np.arange(12)
n_reshape = n.reshape(3, 4)
t_view, t_reshape.shape, t_unsq0.shape, t_unsq1.shape, t_sq.shape, n_reshape.shape


# Broadcasting

# In[6]:


np_b1 = np.array([[1], [2], [3]])
np_b2 = np.array([10, 20, 30])
np_b1 + np_b2
torch_b1 = torch.tensor([[1], [2], [3]])
torch_b2 = torch.tensor([10, 20, 30])
torch_b1 + torch_b2


# In place vs Out of place Operations

# In[7]:


torch_x = torch.tensor([1, 2, 3])
torch_out = torch_x + 5
torch_in = torch_x.clone()
torch_in.add_(5)
np_x = np.array([1, 2, 3])
np_out = np_x + 5
np_in = np_x.copy()
np_in += 5
torch_x, torch_out, torch_in, np_x, np_out, np_in

