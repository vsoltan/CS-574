#!/home/vsoltan/anaconda3/envs/viscomp/bin/python

import torch

print(torch.cuda.is_available())
print(torch.cuda.device(0))
