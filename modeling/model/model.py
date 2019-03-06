import torch
import torch.nn as nn
# import torch.nn.functional as F
from base import BaseModel


# class MnistModel(BaseModel):
#     def __init__(self, num_classes=10):
#         super(MnistModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, num_classes)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

class lwunet(BaseModel):
    def __init__(self, n_grams=1, loops=1):
        super(lwunet, self).__init__()
        # self.encoder = nn.Sequential(
        #     nn.Conv3d(2+n_grams,16,3,stride=1,padding=1),
        #     nn.ReLU(True),
        #     nn.MaxPool3d(2,stride=2),
        #     nn.Conv3d(16,32,3,stride=1,padding=1),
        #     nn.ReLU(True),
        #     nn.MaxPool3d(2,stride=2)
        # )
        self.c1 = nn.Conv3d(2+n_grams,16,3,stride=1,padding=1)
        self.c2 = nn.Conv3d(16,32,3,stride=1,padding=1)
        self.rc1 = nn.ConvTranspose3d(32, 32, 4, 2, 1)
        self.rc2 = nn.ConvTranspose3d(64, 16, 4, 2, 1)
        self.pool = nn.MaxPool3d(2,stride=2)
        self.relu = nn.ReLU(True)
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose3d(32, 32, 4, 2, 1),
        #     nn.ReLU(True),
        #     nn.ConvTranspose3d(32, 16, 4, 2, 1),
        #     nn.ReLU(True)
        # )
        self.logits = nn.Conv3d(32,1,1,1)
        self.variance = nn.Conv3d(32,1,1,1)
        self.softplus = nn.Softplus()
        self.loops = loops
    def forward(self, img, seed, confidence, prev=None):
        """
        img = nx1x64x64x64 images [0,255], dim 1 not required
        seed = nx1x64x64x64 binary, dim 1 not required
        confidence = nx1x64x64x64 [1:inf), dim 1 not required
        prev = nxmx64x64x64 binary, dim 1 required, this is n gram
        """
        if img.dim() == 4:
            img = img.unsqueeze(1)
        if seed.dim() == 4:
            seed = seed.unsqueeze(1)
        if confidence.dim() == 4:
            confidence = confidence.unsqueeze(1)
        for _ in range(self.loops):
            if prev is not None:
                combined = torch.cat((img,prev,seed,confidence), 1)
            else:
                combined = torch.cat((img,seed,confidence), 1)
            x1 = self.c1(combined) #16

            x2 = self.c2(self.pool(self.relu(x1))) #32

            rx1 = self.rc1(self.pool(self.relu(x2))) #32

            rx2 = self.relu(rx1)
            rx2 = self.rc2(torch.cat([x2, rx1], dim=1)) #48

            out_prelim = self.relu(rx2)

            logits = self.logits(torch.cat([x1, out_prelim], dim=1))
            sigma = self.softplus(self.variance(torch.cat([x1, out_prelim], dim=1))).add(1)
            confidence = sigma
            seed = logits.div(sigma).sigmoid()
            #release memory
            x1=None
            x2=None
            rx1=None
            rx2=None
            out_prelim=None
        return logits, sigma