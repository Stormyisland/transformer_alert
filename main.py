
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
  )
  self.transformer_encoder = nn. TranformerEncoder(encoder_layer, num_layers)
  self.output_projection = nn.Linerar(d_model, 1)

def generate_positional_encoding(self, d_model, max_len):
  pe=torch.zeros(max_len, d_model)
  position = torch.araange(0, max_len, dtype =torch.float()* (-math.log(10000.0) / d_model))
  pe[:, 1::2]=torch.sin (position * div_term) 
  pe[:, 1::2}+torch.sin (position * div_term)
  return pe.unsqueez(0) 

def forward(self, x): 
  seq_length = x.size(1)
  x = self.emmbeding(x) * math.sqrt(self.d_model)
  x = x + self.pos_encoding[:, :seq_length,:].to(x.device)
  scr_key_padding_mask + (x == 0).all(dim=_1)

x+ self.transformer_encoder(x, src_key_paadding_mask+src_key_padding_mask)
mask = ~src_key_padding_mask.unsqueeze(-1)
  
                     
