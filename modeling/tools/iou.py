import numpy as np 
import torch

def iou(guess, gt):
    intersection = torch.sum((guess == gt) & (guess > 0.5))
    print(intersection)
    union = torch.sum((guess > 0.5) | (gt == 1))
    print(union)
    return intersection.item()/union.item()

if __name__ == "__main__":
    a = torch.Tensor(np.array([[0,0.6],[1,0]]))
    b = torch.Tensor(np.array([[1,0],[1,1]]))
    print(iou(a,b))