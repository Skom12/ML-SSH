#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from pathlib import Path

import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import f1_score  


# In[2]:


class classifier(nn.ModuleList):
    def __init__(self):
        super(classifier, self).__init__()
        self.seq_len = 200
        self.num_words = 14986772

        
        self.embedding_size = 32
        self.out_size = 256
        
        self.dropout = nn.Dropout(0.5)
        
        self.kernel_1=2
        self.kernel_2=4
        self.kernel_3=8
        self.kernel_4 =10 
        
        self.stride = 2
          
        self.embedding = nn.Embedding(self.num_words+1, self.embedding_size)
        
        self.conv_1 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)
        self.conv_2 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_2, self.stride)
        self.conv_3 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_3, self.stride)
        self.conv_4 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_4, self.stride)


        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
        self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)

        self.fc = nn.Linear(self.final_len(), 1)

    def final_len(self):
        out_conv_1 = ((self.embedding_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1)
        out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_pool_1 = math.floor(out_pool_1)
      
        out_conv_2 = ((self.embedding_size - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_conv_2 = math.floor(out_conv_2)
        out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_pool_2 = math.floor(out_pool_2)
      
        out_conv_3 = ((self.embedding_size - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_conv_3 = math.floor(out_conv_3)
        out_pool_3 = ((out_conv_3 - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_pool_3 = math.floor(out_pool_3)

        out_conv_4 = ((self.embedding_size - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
        out_conv_4 = math.floor(out_conv_4)
        out_pool_4 = ((out_conv_4 - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
        out_pool_4 = math.floor(out_pool_4)
        
        return (out_pool_1 + out_pool_2 + out_pool_3 + out_pool_4)  * self.out_size

    def forward(self, x):
        
        x = self.embedding(x)
        x1 = self.conv_1(x)
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1)

        
        x2 = self.conv_2(x)
        x2 = torch.relu((x2)) 
        x2 = self.pool_2(x2)
        
        x3 = self.conv_3(x)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)

        x4 = self.conv_4(x)
        x4 = torch.relu(x4)
        x4 = self.pool_4(x4)
        
        union = torch.cat((x1, x2, x3, x4), 2)
        union= union.reshape(union.size(0), -1)
        out = self.fc(union)
        out = self.dropout(out)
        out = torch.sigmoid(out)
      
        return out.squeeze()
    
        
        


# In[3]:


model= classifier()

