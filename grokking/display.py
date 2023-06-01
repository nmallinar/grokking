import os
import sys
from math import ceil
import torch
from tqdm import tqdm
import wandb

from data import get_data
from model import Transformer, FCN
from rfm import main as rfm_main
import matplotlib.pyplot as plt
import torch.nn.functional as F

# train_loader, val_loader, context_len = get_data(
#         "x/y",
#         97,
#         0.5,
#         512
#         )
# model = FCN(
#             dim_model=128,
#             num_tokens=99,
#             num_layers=2,
#             hidden_width=512,
#             context_len=context_len
#         )
# optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=1e-3,
#         betas=(0.9, 0.98),
#         weight_decay=1
#         )
checkpoint5 = torch.load("saves/epoch5.pt")
checkpoint320 = torch.load("saves/epoch320.pt")
checkpoint500 = torch.load("saves/epoch500.pt")
lst = checkpoint5['model_state_dict']
index = 0
for i in lst:
    if index % 2 == 1:
        print(str(i))
        print(lst[i].shape)
        l = lst[i]
        lt = torch.transpose(lst[i], 0, 1)
        lm = torch.matmul(lt, l)
        # print(lm.tolist())
        M = lm.detach().numpy()
        plt.clf()
        plt.imshow(M)
        plt.colorbar()
        plt.savefig(f"saves/epoch_5_{index}.pdf")
    index += 1
    
lst = checkpoint320['model_state_dict']
index = 0
for i in lst:
    if index % 2 == 1:
        print(str(i))
        print(lst[i].shape)
        l = lst[i]
        lt = torch.transpose(lst[i], 0, 1)
        lm = torch.matmul(lt, l)
        # print(lm.tolist())
        M = lm.detach().numpy()
        plt.clf()
        plt.imshow(M)
        plt.colorbar()
        plt.savefig(f"saves/epoch_320_{index}.pdf")
    index += 1

lst = checkpoint500['model_state_dict']
index = 0
for i in lst:
    if index % 2 == 1:
        print(str(i))
        print(lst[i].shape)
        l = lst[i]
        lt = torch.transpose(lst[i], 0, 1)
        lm = torch.matmul(lt, l)
        # print(lm.tolist())
        M = lm.detach().numpy()
        plt.clf()
        plt.imshow(M)
        plt.colorbar()
        plt.savefig(f"saves/epoch_500_{index}.pdf")
    index += 1

# plt.imshow()

#model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# print("epoch: " + str(epoch))   
