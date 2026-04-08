#!/usr/bin/env python
# coding: utf-8

# Data Loading and Preprocessing

# poems.csv has 100 poems we will use that for the training

# In[2]:


get_ipython().system('curl -L "https://drive.google.com/uc?export=download&id=1ko4cu8nWhtoQ3OoTqkJdesoq3GtiToev" -o poems.csv')


# In[3]:


get_ipython().system('ls')


# In[4]:


get_ipython().system('uv pip install polars')


# In[5]:


import polars as pl
df = pl.read_csv("poems.csv")


# In[6]:


print(df)


# In[7]:


tokens = df.select(
    pl.col("text")
    .str.to_lowercase()
    .implode()                          
    .list.join(" ")                     
    .str.extract_all(r"\w+|[^\w\s]")    
    .explode()
    .unique()                      
    .alias("tokens")
)


# In[ ]:


full_text = df.select(
    pl.col("text")
    .str.to_lowercase()
    .implode()                          
    .list.join(" "))["text"].to_list()[0]                   
print(full_text)


# In[8]:


import re
token_list = tokens['tokens'].to_list()
#add unknown tokens
token_list = ["<UNK>"] + token_list
word_to_index = {word: idx for idx, word in enumerate(token_list)}
index_to_word = {idx: word for idx, word in enumerate(token_list)}


# In[11]:


words = re.findall(r"\w+|[^\w\s]", full_text.lower())
encoded = [word_to_index.get(w, word_to_index["<UNK>"]) for w in words]


# We use a sliding window approach to construct the dataset

# In[12]:


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

SEQ_LEN = 30  
class TextDataset(Dataset):
    def __init__(self, encoded, seq_length):
        self.encoded = encoded
        self.seq_length = seq_length
    def __len__(self):
        return len(self.encoded) - self.seq_length
    def __getitem__(self, idx):
        x = torch.tensor(self.encoded[idx : idx + self.seq_length])
        y = torch.tensor(self.encoded[idx + 1 : idx + self.seq_length + 1])
        return x, y


# In[22]:


dataset = TextDataset(encoded, SEQ_LEN)
loader = DataLoader(dataset, batch_size=64, shuffle=True)


# In[13]:


len(token_list)


# Model training

# In[14]:


from torch import nn
#Simple Rnn
class OneHotRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.r = nn.RNN(input_size,hidden_size,batch_first=True)
        self.f = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        seqs = nn.functional.one_hot(x,self.input_size).float()
        outputs,_ = self.r(seqs)
        logits = self.f(outputs) 
        return logits




# In[15]:


#Embedding RNN
class EmbedRNN(nn.Module):
    def __init__(self, input_size,embedding_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.e = torch.nn.Embedding(input_size,embedding_size)
        self.r = nn.RNN(embedding_size,hidden_size,batch_first=True)
        self.f = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        seqs = self.e(x).float()
        outputs,_ = self.r(seqs)
        logits = self.f(outputs) 
        return logits




# In[16]:


from torch import nn
#Simple LSTM
class OneHotLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.r = nn.LSTM(input_size,hidden_size,batch_first=True)
        self.f = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        seqs = nn.functional.one_hot(x,self.input_size).float()
        outputs,_ = self.r(seqs)
        logits = self.f(outputs) 
        return logits




# In[17]:


#Embedding LSTM
class EmbedLSTM(nn.Module):
    def __init__(self, input_size,embedding_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.e = torch.nn.Embedding(input_size,embedding_size)
        self.r = nn.LSTM(embedding_size,hidden_size,batch_first=True)
        self.f = nn.Linear(hidden_size,output_size)
    def forward(self,x):
        seqs = self.e(x).float()
        outputs,_ = self.r(seqs)
        logits = self.f(outputs) 
        return logits




# Training

# In[24]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")


# In[25]:


def train_model(model,loader,epochs):
 model.to(device)
 optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 for epoch in range(epochs):
  total_loss = 0
  for batch in loader:
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    logits = model(inputs)
    logits = logits.view(-1,logits.size(-1))
    targets = targets.view(-1)
    loss = nn.CrossEntropyLoss()(logits, targets)
    total_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print("Epoch",epoch,"Loss",total_loss/len(loader))


# In[29]:


oneRNN = OneHotRNN(input_size=5177, hidden_size=128, output_size=5177)
train_model(oneRNN, loader, epochs=20)


# In[33]:


embedRNN = EmbedRNN(input_size=5177, embedding_size = 64,hidden_size=128, output_size=5177)
train_model(embedRNN, loader, epochs=30)


# In[36]:


oneLSTM = OneHotLSTM(input_size=5177, hidden_size=128, output_size=5177)
train_model(oneLSTM, loader, epochs=50)


# In[37]:


embedLSTM = EmbedLSTM(input_size=5177, embedding_size=64, hidden_size=128, output_size=5177)
train_model(embedLSTM, loader, epochs=50)


# In[34]:


def generate(model, start_text, length=50):
    model.eval()
    words = re.findall(r"\w+|[^\w\s]", start_text.lower())
    input_ids = [word_to_index.get(w, word_to_index["<UNK>"]) for w in words]

    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([input_ids[-SEQ_LEN:]]).to(device)
            logits = model(x)
            next_id = logits[0, -1].argmax().item()
            input_ids.append(next_id)

    return " ".join(index_to_word[i] for i in input_ids)


# In[40]:


print(generate(oneRNN, "the son smiles", length=30))
print(generate(embedRNN, "the violet is", length=30))
print(generate(oneLSTM, "the son smiles", length=30))
print(generate(embedLSTM, "the violet is", length=30))


# We sucessfully trained the models on the poems dataset and were able to get the outputs for the text, One hot encoding models generally perform a bit worse than the embedded ones in both RNN and LSTM cases
