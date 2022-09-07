#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import f1_score  


# In[5]:


class classifier(nn.ModuleList):
    def __init__(self):
        super(classifier, self).__init__()
        self.num_words = 7584901
        self.embedding_size =64
        self.seq_len = 200
        self.hidden_dim=128
        
        self.dropout=nn.Dropout(0.5)
        self.embeddings=nn.Embedding(self.num_words+1, self.embedding_size)
        self.flatten=nn.Flatten()
        
        self.kernel_1=2
        self.kernel_2=4

        
        
        self.stride = 2
        
        self.conv1 =nn.Conv1d(self.seq_len, self.hidden_dim, self.kernel_1, self.stride)
        self.conv2 =nn.Conv1d(self.hidden_dim, self.hidden_dim, self.kernel_2, self.stride)

        
        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        
        self.fc1 = nn.Linear(256, 1)
        


        
    def forward(self,x):
        

        x=self.embeddings(x)

        out=self.conv1(x)
        out=torch.relu(out)
        out=self.pool_1(out)
        out=self.dropout(out)

        out=self.conv2(out)
        out=torch.relu(out)
        out=self.pool_2(out)
        out=self.dropout(out)
        
        out=self.flatten(out)

        out=self.fc1(out)
     
        out=torch.sigmoid(out)

        return out.squeeze()
        
        
        
        
        
        
        


# In[3]:


model=classifier()


# In[ ]:




