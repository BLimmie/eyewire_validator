from  models.custom_loss import Bayes_BCE_Loss_With_Logits
import torch
from tools.create_data import create_gt

x = torch.randn(256,256,256, requires_grad=True)
sigma = torch.ones(256,256,256, dtype=torch.float, requires_grad=True)
y = torch.tensor(create_gt(10531), dtype=torch.float)

loss_fn = Bayes_BCE_Loss_With_Logits.apply

loss = loss_fn(x,y,sigma)
print(loss.item())
loss.backward()
print(x.grad)
print(sigma.grad)