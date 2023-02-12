import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch.cuda.amp import autocast, GradScaler
