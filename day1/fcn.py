import os 
import torch 
import tiktoken
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn


CONTEXT_LENGTH = 8 
EMB_DIM = 768
VOCAB_SIZE = 50257
BATCH_SIZE = 4
STRIDE = 4

tokenizer = tiktoken.get_encoding('gpt2')
data = 'Ola, me chamo fernando, como voce esta? Baum dmais? Vamo fazer alguma coisa ai po tamo junto. Cara vamo la, hoje em dia as coisas nao esao mais desse jeito, vamo compreender ai po! Voce nao queria ter uma vida mais dboa fazendo oque voce gosta? Eh so querer, se apaixonar pela parada, quando mais fazer nao interessa quanto tempo mais rapido chega la'
token_ids = tokenizer.encode(data)



def simple_attention(h):
    #weights and context 
    scores = h @ h.transpose(-2, -1) # why do we do this? what does score represents
    weigths = torch.softmax(scores, dim=-1)
    context = weigths @ h # multiplaying matrices?
    return context, weigths

def create_chunks(token_ids, ctx_length, stride):
    inputs, targets = [], []
    for i in range(0, len(token_ids) - ctx_length, stride):
        inputs.append(token_ids[i: i + ctx_length])
        targets.append(token_ids[i + 1: i + ctx_length + 1])
        print(f"{tokenizer.decode(token_ids[i: i + ctx_length])} --> {tokenizer.decode(token_ids[i+1: i + ctx_length +1])}")
        #print(f"{inputs} --> {targets}")
        
    return torch.tensor(inputs), torch.tensor(targets)

print(f"Token ids are: {token_ids}")
# number of chunks ≈ (N - CONTEXT_LENGTH) / STRIDE = (N - 8) / 4
inputs, targets = create_chunks(token_ids, CONTEXT_LENGTH, STRIDE) # what is the impact on increasing ctx_length or 
                                                 # stride, what is the math behind it?
dataset = TensorDataset(inputs, targets)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

tok_emb = nn.Embedding(VOCAB_SIZE, EMB_DIM) # token table [50257, 768]
pos_emb = nn.Embedding(CONTEXT_LENGTH, EMB_DIM) #256, 768

d_in, d_out = EMB_DIM, 64
torch.manual_seed(123)

all_context_vecs = []
for x, y in loader:
    tok_vec = tok_emb(x)
    pos_vec = pos_emb(torch.arange(x.shape[1])) # what is the x.shape doing?
                                                # Random initial numbers
    inp_vec = tok_vec + pos_vec
    context, weigths = simple_attention(inp_vec)
    all_context_vecs.append(context)

all_context_vecs = torch.cat(all_context_vecs)
print("Context vectors: ", all_context_vecs.shape)
print("Row sums of weigths: ", weigths[0].sum(dim=-1)[:5])
print("weight each token gives itself: ", weigths.diagonal(dim1=-2, dim2=-1).mean().item()) 


# dataset
# dataloader



