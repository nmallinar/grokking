from einops import rearrange, repeat
import torch
from torch import nn, Tensor
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)

class FCNEmbedded(torch.nn.Module):
  def __init__(self, dim_model: int, num_tokens: int, num_layers: int, hidden_width: int, context_len: int):
    super().__init__()

    self.num_tokens = num_tokens
    layers = []
    inp_dim = dim_model * context_len
    for idx in range(num_layers-1):
        layers.append(nn.Linear(inp_dim, hidden_width))
        layers.append(nn.ReLU())
        inp_dim = hidden_width

    layers.append(nn.Linear(inp_dim, num_tokens))

    self.layers = nn.Sequential(*layers)

  def forward(self, x: Tensor):
    return self.layers(x)

class TwoLayerFCN(torch.nn.Module):
  def __init__(self, dim_model: int, num_tokens: int, hidden_width: int, context_len: int):
    super().__init__()

    self.num_tokens = num_tokens
    inp_dim = dim_model * context_len
    inp_dim = self.num_tokens * context_len
    self.inp_dim = inp_dim
    self.hidden_width = hidden_width

    self.fc1 = nn.Linear(inp_dim, hidden_width, bias=False)
    self.fc2 = nn.Linear(hidden_width, hidden_width, bias=False)
    self.out = nn.Linear(hidden_width, num_tokens, bias=False)

  def forward(self, x, dumb1=None, dumb2=None, dumb3=None, return_hid=False):
      if dumb1 is None:
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          if return_hid:
              return x
          return self.out(x)

      x = F.relu(self.fc1(x) + dumb1 @ self.fc1.weight.t())
      x = F.relu(self.fc2(x) + dumb2 @ self.fc2.weight.t())
      if return_hid:
          return x
      x = self.out(x) + dumb3 @ self.out.weight.t()
      return x

class FCN(torch.nn.Module):
  def __init__(self, dim_model: int, num_tokens: int, num_layers: int, hidden_width: int, context_len: int):
    super().__init__()

    self.num_tokens = num_tokens
    layers = []

    inp_dim = dim_model * context_len
    for idx in range(num_layers):
        layers.append(nn.Linear(inp_dim, hidden_width, bias=False))
        layers.append(nn.ReLU())
        inp_dim = hidden_width

    layers.append(nn.Linear(inp_dim, num_tokens, bias=False))
    self.layers = nn.Sequential(*layers)

  def forward(self, x: Tensor):
    for layer in self.layers:
        x = layer(x)
    return x

class OldFCN(torch.nn.Module):
  def __init__(self, dim_model: int, num_tokens: int, num_layers: int, hidden_width: int, context_len: int):
    super().__init__()

    self.num_tokens = num_tokens
    # self.token_embeddings = nn.Embedding(num_tokens, dim_model)
    # self.token_embeddings.requires_grad_(False)
    layers = []
    layers += [
        nn.Linear(num_tokens, dim_model, bias=False),
        nn.Flatten()
    ]

    inp_dim = dim_model * context_len
    for idx in range(num_layers-1):
        layers.append(nn.Linear(inp_dim, hidden_width, bias=False))
        layers.append(nn.ReLU())
        inp_dim = hidden_width

    layers.append(nn.Linear(inp_dim, num_tokens, bias=False))
    #self.layers = layers
    self.layers = nn.Sequential(*layers)

  def forward(self, x: Tensor, return_hid=False):
    # batch_size = x.shape[0]

    # token_embedding = self.token_embeddings(x)
    # token_embedding = token_embedding.view(batch_size, -1)
    # return self.layers(token_embedding)
    # return self.layers(x)
    x = F.one_hot(x.long(), num_classes=self.num_tokens).double()
    hid = [x]
    x = self.layers[0](x)
    x = self.layers[1](x)
    hid.append(x)
    for layer in self.layers[2:]:
        x = layer(x)
        hid.append(x)
    return tuple(hid)
    if return_hid:
        return hid
    return x

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
