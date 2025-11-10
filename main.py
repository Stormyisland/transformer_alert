
import torch
import torch.nn as nn
import math

class SimpleTransformer(nn.Module):
  def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, max_seq_length=512):
  super(simpleTransformer, self).__init__()
  self.d_model = d_model
  self.embedding = nn.Embedding(vocab_size, d_model)
  self.pos_encodeing = self._generate_positional_encodeing(d_model, max_seq_length)

  encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=2048,
    droopout=0.1,
    batch_first=True
