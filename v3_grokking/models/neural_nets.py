import numpy as np
from einops import rearrange, repeat
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import scipy
import math

torch.set_default_dtype(torch.float64)

class OneLayerFCN(torch.nn.Module):
  def __init__(self, num_tokens: int, hidden_width: int,
               context_len: int, init_scale=1.0, n_classes=-1):
    super().__init__()

    if n_classes == -1:
        n_classes = num_tokens

    self.num_tokens = num_tokens
    inp_dim = self.num_tokens * context_len
    self.inp_dim = inp_dim
    self.hidden_width = hidden_width
    self.init_scale = init_scale

    self.fc1 = nn.Linear(inp_dim, hidden_width, bias=False)
    #self.fc2 = nn.Linear(hidden_width, hidden_width, bias=False)
    #self.fc3 = nn.Linear(hidden_width, hidden_width, bias=False)
    #self.fc4 = nn.Linear(hidden_width, hidden_width, bias=False)
    self.out = nn.Linear(hidden_width, n_classes, bias=False)

    self.reset_params(init_scale=init_scale)

  def reset_params(self, init_scale=1.0):
    #kernel_M = torch.from_numpy(np.load('x+y_M.npy')).float()
    #dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(self.fc1.weight.shape[1]), kernel_M + 1e-6 * torch.eye(kernel_M.shape[0]))
    #self.fc1.weight.data = dist.sample_n(self.fc1.weight.shape[0])

    # scaled kaiming uniform code:
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(self.fc1.weight, -init_scale*bound, init_scale*bound)

    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.out.weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(self.out.weight, -init_scale*bound, init_scale*bound)

    #nn.init.uniform_(self.fc2.weight, -init_scale*bound, init_scale*bound)
    #nn.init.uniform_(self.fc3.weight, -init_scale*bound, init_scale*bound)
    #nn.init.uniform_(self.fc4.weight, -init_scale*bound, init_scale*bound)

    # scaled kaiming normal code:
    '''
    leaky_neg_slope = 0.
    fan = nn.init._calculate_correct_fan(self.fc1.weight, "fan_in")
    gain = nn.init.calculate_gain("leaky_relu", leaky_neg_slope)
    std = gain/math.sqrt(fan)
    nn.init.normal_(self.fc1.weight, mean=0.0, std=init_scale*std)

    fan = nn.init._calculate_correct_fan(self.out.weight, "fan_in")
    gain = nn.init.calculate_gain("leaky_relu", leaky_neg_slope)
    std = gain/math.sqrt(fan)
    nn.init.normal_(self.out.weight, mean=0.0, std=init_scale*std)
    '''
    #w1 = torch.from_numpy(np.load('outdir/cluster_covariances/ep_2800_w1_p31_relu.npy')).double()
    #nfm = w1.T @ w1
    #nfm = torch.from_numpy(np.load('nfm_scale1e-4.npy')).double()
    #dist = torch.distributions.multivariate_normal.MultivariateNormal(
    #    torch.zeros(self.fc1.weight.shape[1]),
    #    nfm
    #)
    #self.fc1.weight.data = dist.sample_n(self.fc1.weight.shape[0])

    #fc1_wvecs = []
    #for idx in range(15):
    #    cov = torch.from_numpy(np.load(f'outdir/cluster_covariances/cov_{idx}.npy')).double()
    #    dist = torch.distributions.multivariate_normal.MultivariateNormal(
    #       torch.zeros(self.fc1.weight.shape[1]),
    #       cov + 0.0 * torch.eye(self.fc1.weight.shape[1])
    #    )
    #    fc1_wvecs.append(dist.sample_n(int(self.fc1.weight.shape[0]/15)))
    #self.fc1.weight.data = torch.concatenate(fc1_wvecs, dim=0)
    #self.fc1.requires_grad_(False)

    self.init_w0 = self.fc1.weight.detach().clone().T

  def get_random_matrix(self, init_scale=1.0):
    w0 = torch.randn(*list(self.fc1.weight.size()), requires_grad=False)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(w0, -init_scale*bound, init_scale*bound)
    return w0.T

  def forward(self, x, dumb1=None, act='relu'):
      if act == 'relu':
          act_fn = F.relu
      elif act == 'swish':
          act_fn = F.silu
      elif act == 'pow2':
          act_fn = lambda x: torch.pow(x, 2)
      elif act == 'softplus':
          act_fn = F.softplus
      elif act == 'linear':
          act_fn = lambda x: x
      elif act == 'hermite2':
          act_fn = lambda x: (torch.pow(x, 2) - 1)/math.sqrt(2)

      if dumb1 is None:
          x = self.fc1(x)
          x = act_fn(x)

          return self.out(x)

      x = act_fn(self.fc1(x) + dumb1 @ self.fc1.weight.t())
      x = self.out(x)
      return x


  '''
  def forward(self, x, dumb1=None, dumb2=None, dumb3=None,
              dumb4=None, return_layer=None, act='relu'):
      if act == 'relu':
          act_fn = F.relu
      elif act == 'swish':
          act_fn = F.silu
      elif act == 'pow2':
          act_fn = lambda x: torch.pow(x, 2)
      elif act == 'softplus':
          act_fn = F.softplus
      elif act == 'linear':
          act_fn = lambda x: x
      elif act == 'hermite2':
          act_fn = lambda x: (torch.pow(x, 2) - 1)/math.sqrt(2)

      if dumb1 is None:
          if return_layer == 'M^.5x' or return_layer == 'act_fn(M^.5x)':
              M = self.fc1.weight.t() @ self.fc1.weight
              L, V = torch.linalg.eigh(M)
              sqrtM = V @ torch.diag(torch.sqrt(L)) @ V.T
              if return_layer == 'M^.5x':
                  return x @ sqrtM
              return act_fn(x @ sqrtM)

          x = self.fc1(x)
          if return_layer == 'lin1':
              return x
          x = act_fn(x)
          if return_layer == 'act_fn(lin1)':
              return x

          return self.out(x)

      x = act_fn(self.fc1(x) + dumb1 + dumb3 @ self.fc1.weight.t())
      x = self.out(x) + dumb2 + dumb4 @ self.out.weight.t()
      return x
  '''

class TwoLayerFCN(torch.nn.Module):
   def __init__(self, num_tokens: int, hidden_width: int,
                context_len: int, init_scale=1.0, n_classes=-1):
     super().__init__()

     if n_classes == -1:
         n_classes = num_tokens

     self.num_tokens = num_tokens
     inp_dim = self.num_tokens * context_len
     self.inp_dim = inp_dim
     self.hidden_width = hidden_width

     self.fc1 = nn.Linear(inp_dim, hidden_width, bias=False)
     self.fc2 = nn.Linear(hidden_width, hidden_width, bias=False)
     self.out = nn.Linear(hidden_width, n_classes, bias=False)

     self.reset_params(init_scale=init_scale)

   def reset_params(self, init_scale=1.0):
     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
     nn.init.uniform_(self.fc1.weight, -init_scale*bound, init_scale*bound)

     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc2.weight)
     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
     nn.init.uniform_(self.fc2.weight, -init_scale*bound, init_scale*bound)

     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.out.weight)
     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
     nn.init.uniform_(self.out.weight, -init_scale*bound, init_scale*bound)


   def forward(self, x, dumb1=None, dumb2=None, dumb3=None,
               dumb4=None, dumb5=None, dumb6=None,
               return_layer=None, act='relu'):
       if act == 'relu':
           act_fn = F.relu
       elif act == 'swish':
           act_fn = F.silu
       elif act == 'pow2':
           act_fn = lambda x: torch.pow(x, 2)
       elif act == 'softplus':
           act_fn = F.softplus
       elif act == 'linear':
           act_fn = lambda x: x

       if dumb1 is None:
           if return_layer == 'M^.5x' or return_layer == 'act_fn(M^.5x)':
               M = self.fc1.weight.t() @ self.fc1.weight
               L, V = torch.linalg.eigh(M)
               sqrtM = V @ torch.diag(torch.sqrt(L)) @ V.T
               if return_layer == 'M^.5x':
                   return x @ sqrtM
               return act_fn(x @ sqrtM)

           x = self.fc1(x)
           if return_layer == 'lin1':
               return x
           x = act_fn(x)
           if return_layer == 'act_fn(lin1)':
               return x
           x = self.fc2(x)
           if return_layer == 'lin2':
               return x
           x = act_fn(x)
           if return_layer == 'act_fn(lin2)':
               return x

           return self.out(x)

       x = act_fn(self.fc1(x) + dumb1 + dumb4 @ self.fc1.weight.t())
       x = act_fn(self.fc2(x) + dumb2 + dumb5 @ self.fc2.weight.t())
       x = self.out(x) + dumb3 + dumb6 @ self.out.weight.t()
       return x

class FourLayerFCN(torch.nn.Module):
   def __init__(self, num_tokens: int, hidden_width: int,
                context_len: int, init_scale=1.0, n_classes=-1):
     super().__init__()

     if n_classes == -1:
         n_classes = num_tokens

     self.num_tokens = num_tokens
     inp_dim = self.num_tokens * context_len
     self.inp_dim = inp_dim
     self.hidden_width = hidden_width

     self.fc1 = nn.Linear(inp_dim, hidden_width, bias=False)
     self.fc2 = nn.Linear(hidden_width, hidden_width, bias=False)
     self.fc3 = nn.Linear(hidden_width, hidden_width, bias=False)
     self.fc4 = nn.Linear(hidden_width, hidden_width, bias=False)
     self.out = nn.Linear(hidden_width, n_classes, bias=False)

     self.reset_params(init_scale=init_scale)

   def reset_params(self, init_scale=1.0):
     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
     nn.init.uniform_(self.fc1.weight, -init_scale*bound, init_scale*bound)

     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc2.weight)
     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
     nn.init.uniform_(self.fc2.weight, -init_scale*bound, init_scale*bound)

     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc3.weight)
     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
     nn.init.uniform_(self.fc3.weight, -init_scale*bound, init_scale*bound)

     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc4.weight)
     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
     nn.init.uniform_(self.fc4.weight, -init_scale*bound, init_scale*bound)

     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.out.weight)
     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
     nn.init.uniform_(self.out.weight, -init_scale*bound, init_scale*bound)


   def forward(self, x, dumb1=None, dumb2=None, dumb3=None,
               dumb4=None, dumb5=None, dumb6=None,
               return_layer=None, act='relu'):
       if act == 'relu':
           act_fn = F.relu
       elif act == 'swish':
           act_fn = F.silu
       elif act == 'pow2':
           act_fn = lambda x: torch.pow(x, 2)
       elif act == 'softplus':
           act_fn = F.softplus
       elif act == 'linear':
           act_fn = lambda x: x

       if dumb1 is None:
           x = self.fc1(x)
           if return_layer == 'lin1':
               return x
           x = act_fn(x)
           if return_layer == 'act_fn(lin1)':
               return x
           x = self.fc2(x)
           if return_layer == 'lin2':
               return x
           x = act_fn(x)
           if return_layer == 'act_fn(lin2)':
               return x
           x = self.fc3(x)
           if return_layer == 'lin3':
               return x
           x = act_fn(x)
           if return_layer == 'act_fn(lin3)':
               return x
           x = self.fc4(x)
           if return_layer == 'lin4':
               return x
           x = act_fn(x)
           if return_layer == 'act_fn(lin4)':
               return x

           return self.out(x)

       x = act_fn(self.fc1(x) + dumb1 + dumb4 @ self.fc1.weight.t())
       x = act_fn(self.fc2(x) + dumb2 + dumb5 @ self.fc2.weight.t())
       x = act_fn(self.fc3(x))
       x = act_fn(self.fc4(x))
       x = self.out(x) + dumb3 + dumb6 @ self.out.weight.t()
       return x
