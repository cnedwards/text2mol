
from torch import nn

import torch

CE = nn.CrossEntropyLoss()

def contrastive_loss(v1, v2):
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
  labels = torch.arange(logits.shape[0], device=v1.device)
  return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

BCEL = nn.BCEWithLogitsLoss()

def negative_sampling_contrastive_loss(v1, v2, labels):
  logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
  eye = torch.diag_embed(labels).to(v1.device)
  return BCEL(logits, eye) + BCEL(torch.transpose(logits, 0, 1), eye), logits.diag() > 0
