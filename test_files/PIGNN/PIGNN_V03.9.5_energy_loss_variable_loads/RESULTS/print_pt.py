import torch
import os
import time
import torch
import numpy as np
from pathlib import Path

CURRENT_SUBFOLDER = Path(__file__).resolve().parent
os.chdir(CURRENT_SUBFOLDER)

data = torch.load("epoch_2000.pt")
data = torch.load("epoch_2000.pt", weights_only=True)
print(type(data))
print(data.keys())
print(data['losses'])