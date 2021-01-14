import torch
import numpy as np
from torch.nn import functional as F

# F = torch.Tensor([[0]])
a = torch.Tensor([[1,2,3]])
b = torch.Tensor([[4,5,6]])
c = torch.Tensor([[7,8,9]])
d = torch.cat((a,b),dim=0)
pred = F.softmax(d, dim=1).cpu().numpy()

print(pred)
print(pred.argmax(1))

# print(a)









