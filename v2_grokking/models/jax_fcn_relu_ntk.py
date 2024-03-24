from neural_tangents import stax
import jax.dlpack
from jax import jacfwd, jacrev
from jax.scipy.linalg import sqrtm

def torch2jax(x):
    x = torch.to_dlpack(x)
    return jax.dlpack.from_dlpack(x)

def jax2torch(x):
    x = jax.dlpack.to_dlpack(x)
    x = torch.utils.dlpack.from_dlpack(x)

def jax_ntk_fn(x, y, M=None, depth=2, bias=0, kernel_fn=None):
    if M is not None:
        sqrt_M = sqrtm(M)
        x = x @ sqrt_M
        y = y @ sqrt_M

    # x = torch.to_dlpack(x)
    # x = jax.dlpack.from_dlpack(x)
    # y = torch.to_dlpack(y)
    # y = jax.dlpack.from_dlpack(y)

    if kernel_fn is None:
        layers = []
        for _ in range(1, depth):
            layers += [
                stax.Dense(1, W_std=1, b_std=bias),
                stax.Relu()
            ]
        layers.append(stax.Dense(1, W_std=1, b_std=bias))

        _, _, kernel_fn = stax.serial(*layers)

    ntk_nngp_jax = kernel_fn(x, y)

    # ntk_jax = jax.dlpack.to_dlpack(ntk_nngp_jax.ntk)
    # ntk_jax = torch.utils.dlpack.from_dlpack(ntk_jax)
    # nngp_jax = jax.dlpack.to_dlpack(ntk_nngp_jax.nngp)
    # nngp_jax = torch.utils.dlpack.from_dlpack(nngp_jax)

    return nngp_jax, ntk_jax
