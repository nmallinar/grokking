from einops import rearrange, repeat
import torch
from torch import nn, Tensor

class FCN(torch.nn.Module):
  def __init__(self, dim_model: int, num_tokens: int, num_layers: int, hidden_width: int, context_len: int):
    super().__init__()

    self.num_tokens = num_tokens
    self.token_embeddings = nn.Embedding(num_tokens, dim_model)
    self.token_embeddings.requires_grad_(False)
    layers = []

    inp_dim = dim_model * context_len
    for idx in range(num_layers-1):
        layers.append(nn.Linear(inp_dim, hidden_width, bias=False))
        layers.append(nn.ReLU())
        inp_dim = hidden_width

    layers.append(nn.Linear(inp_dim, num_tokens, bias=False))

    self.layers = nn.Sequential(*layers)
  # def __init__(self, dim_model: int, num_tokens: int, num_layers: int, hidden_width: int, context_len: int):
  #   super().__init__()

  #   self.token_embeddings = nn.Embedding(num_tokens, dim_model)
  #   layers = []

  #   if num_layers > 1:
  #       layers.append(nn.Linear(dim_model*context_len, hidden_width))
  #       layers.append(nn.ReLU())
  #       for _ in range(num_layers-1):
  #           layers.append(nn.Linear(hidden_width, hidden_width))
  #           layers.append(nn.ReLU())
  #       layers.append(nn.Linear(hidden_width, num_tokens))
  #   else:
  #       layers.append(nn.Linear(dim_model*context_len, num_tokens))

  #   self.layers = nn.Sequential(*layers)

  def forward(self, x: Tensor):
    batch_size, context_len = x.shape

    token_embedding = self.token_embeddings(x)
    token_embedding = token_embedding.view(batch_size, -1)
    return self.layers(token_embedding)

class DecoderBlock(torch.nn.Module):
  def __init__(self, dim_model: int, n_heads: int):
    super().__init__()

    self.self_attn = nn.MultiheadAttention(dim_model, n_heads)
    self.self_attn_norm = nn.LayerNorm(dim_model)
    self.ffn = nn.Sequential(
        nn.Linear(dim_model, dim_model * 4),
        nn.GELU(),
        nn.Linear(dim_model * 4, dim_model)
    )
    self.ffn_norm = nn.LayerNorm(dim_model)

  def forward(self, x: Tensor):
    attn_mask = torch.full(
        (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
    )
    attn_mask = torch.triu(attn_mask, diagonal=1)

    a1, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
    a1 = self.self_attn_norm (x + a1)
    a2 = self.ffn(a1)
    a2 = self.ffn_norm(a1 + a2)

    return a2

class Transformer(torch.nn.Module):
  def __init__(self, num_layers: int, dim_model: int, num_heads: int, num_tokens: int, seq_len: int):
    super().__init__()

    self.token_embeddings = nn.Embedding(num_tokens, dim_model)
    self.position_embeddings = nn.Embedding(seq_len, dim_model)
    self.model = nn.Sequential(
        *[DecoderBlock(dim_model, num_heads) for _ in range(num_layers)],
        nn.LayerNorm(dim_model),
        nn.Linear(dim_model, num_tokens)
    )

  def forward(self, inputs: Tensor):
    batch_size, context_len = inputs.shape

    token_embedding = self.token_embeddings(inputs)

    positions = repeat(torch.arange(context_len, device=inputs.device), "p -> b p", b = batch_size)
    position_embedding = self.position_embeddings(positions)

    embedding = token_embedding + position_embedding

    embedding = rearrange(embedding, 'b s d -> s b d')

    out = self.model(embedding)
    return out[-1, :, :]
