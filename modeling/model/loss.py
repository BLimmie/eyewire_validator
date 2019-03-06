import torch
from torch.autograd import Function as Function

class Bayes_BCE_Loss_With_Logits(Function):
    @staticmethod
    def forward(ctx, x, y, sigma):
        ctx.save_for_backward(x,y,sigma)
        p = x.div(sigma)
        bce_loss_logits = torch.nn.BCEWithLogitsLoss(reduction='none')
        bce_loss = bce_loss_logits(p, y)
        sigma_loss = sigma.log()
        #print(bce_loss)
        total_loss = bce_loss.add(1, sigma)
        return total_loss.mean()
    
    @staticmethod
    def backward(ctx, grad_output):
        x,y,sigma = ctx.saved_tensors
        z = x.div(sigma)
        dx = ((((
                y.add(-1)
                ).mul(z.exp())
                ).add(y)
                ).div(
                    sigma.mul(z.exp().add(1))
                )
            ).mul(-1)
        dy = grad_output
        d_sigma = (z.exp().mul(x.mul(y.add(-1)).add(1,sigma)).add(x.mul(y)).add(1,sigma)).div(sigma.pow(2).mul(z.exp().add(1)))
        return dx,dy,d_sigma

loss_fn = Bayes_BCE_Loss_With_Logits.apply

def Bayes_BCE_Logits_Loss(x,y,sigma):
    return loss_fn(x,y,sigma)