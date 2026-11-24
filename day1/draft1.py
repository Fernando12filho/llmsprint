import torch 

"""
    Implementation of how attention is calculated
"""

inputs = torch.tensor((
    [[0.43, 0.15, 0.14], 
     [0.21, 0.57, 0.57], 
     [0.67, 0.24, 0.15]]
    )) 
 
query = inputs[1] 
attn_scores_2 = torch.empty(inputs.shape[0])  
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)

attn_weigths_2_temp = attn_scores_2 / attn_scores_2.sum()
print(f"Attention Weights are: {attn_weigths_2_temp}")
print(f"Sum of attention weights is: {attn_weigths_2_temp.sum()}")






