from models.autoencoder import autoencoder
from tools.create_data import create_stack
from models.custom_loss import Bayes_BCE_Loss_With_Logits
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = autoencoder()
model = model.to(device)
img,seed,conf,gt = create_stack(10531)
img = torch.tensor(img, dtype=torch.float, requires_grad=True).unsqueeze(0)
img = img.to(device)
seed = torch.tensor(seed, dtype=torch.float, requires_grad=True).unsqueeze(0)
seed = seed.to(device)
conf = torch.tensor(conf, dtype=torch.float, requires_grad=True).unsqueeze(0)
conf = conf.to(device)
gt = torch.tensor(gt, dtype=torch.float).unsqueeze(0).unsqueeze(1)
gt = gt.to(device)

#loss_fn = torch.nn.BCEWithLogitsLoss()
loss_fn = Bayes_BCE_Loss_With_Logits.apply

logits, sigma = model(img,seed,conf)
print("Model ran")
loss = loss_fn(logits, gt, sigma)
print("Loss forward ran")
#print(loss)
loss.backward()
print("Loss backward ran")