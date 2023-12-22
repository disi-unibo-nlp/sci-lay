import jax
import torch
from jax2torch import jax2torch
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sparse_soft_topk
# Define input values
values = torch.tensor([-5., -2., 3., 1.]).to("cuda:0")
torch_fun = jax2torch(sparse_soft_topk.sparse_soft_topk_mask_pav)
# First, we can return the top-2 mask of the vector values
print("soft top-k mask with PAV: ", torch_fun(values, k=2, l=1e-2))