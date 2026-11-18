import os
import torch 
import tiktoken
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

VOCAB_SIZE = 50257
EMB_DIM = 768
CTX_LENGTH = 1024
tokenizer = tiktoken.get_encoding("gpt2")
with open("input.txt", "r") as f:
    data = f.read()
"""
words = data.splitlines()
words = [w.strip() for w in words] # rid of any leading or triling white space
words = [w for w in words if w] # get rid of any empty strings
chars = sorted(list(set(''.join(words)))) # all the possible characters
max_word_length = max(len(w) for w in words)
print(f"number of examples in the dataset: {len(words)}")
print(f"max word length: {max_word_length}")
print(f"number of unique characters in the vocabulary: {len(chars)}")
print("Vocabulary:")
print(''.join(chars))
"""

"""
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate (chars)}

def encode(s: str) -> list[int]:
	return [stoi[c] for c in s]

def decode(tokens: list[int]) -> str:
	return "".join([itos[i] for i in tokens])
"""

token_ids = tokenizer.encode(data)
ids = tokenizer.encode("ROMEO:")
text_out = tokenizer.decode(ids)

print(f"Token ids for word {text_out} is {ids}")

def make_chunks(token_ids, max_length, stride):
    inputs, targets = [], []
    for i in range(0, len(token_ids) - max_length, stride):
        inputs.append(token_ids[i: i+max_length])
        targets.append(token_ids[i+1: i + max_length + 1])
    return torch.tensor(inputs), torch.tensor(targets)

inputs, targets = make_chunks(token_ids, 256, 4)
dataset = TensorDataset(inputs, targets)
loader = DataLoader(dataset, batch_size = 4, shuffle = True, drop_last = True)
tok_emb = nn.Embedding(VOCAB_SIZE, EMB_DIM)
pos_emb = nn.Embedding(CTX_LENGTH, EMB_DIM)

x, y = next(iter(loader))
tok_vector = tok_emb(x)
pos_vector = pos_emb(torch.arange(256))

