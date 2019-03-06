import torch
import numpy as np

# def my_metric(output, target):
#     with torch.no_grad():
#         pred = torch.argmax(output, dim=1)
#         assert pred.shape[0] == len(target)
#         correct = 0
#         correct += torch.sum(pred == target).item()
#     return correct / len(target)

# def my_metric2(output, target, k=3):
#     with torch.no_grad():
#         pred = torch.topk(output, k, dim=1)[1]
#         assert pred.shape[0] == len(target)
#         correct = 0
#         for i in range(k):
#             correct += torch.sum(pred[:, i] == target).item()
#     return correct / len(target)

def precision(output, sigma, target, threshold=0.5):
    with torch.no_grad():
        assert output.shape == target.shape
        guess = output.div(sigma).sigmoid()
        intersection = torch.sum((target == 1) & (guess > threshold)).item()
        total_guesses = torch.sum(guess > threshold).item()
    if total_guesses == 0:
        return 0.0
    return intersection / total_guesses

def recall(output, sigma, target, threshold=0.5):
    with torch.no_grad():
        assert output.shape == target.shape
        guess = output.div(sigma).sigmoid()
        intersection = torch.sum((target == 1) & (guess > threshold)).item()
        total_gt = torch.sum(target == 1).item()
    if total_gt == 0:
        return 1.0
    return intersection / total_gt

def iou(output, sigma, target, threshold=0.5):
    with torch.no_grad():
        assert output.shape == target.shape
        guess = output.div(sigma).sigmoid()
        intersection = torch.sum((target == 1) & (guess > threshold)).item()
        union = torch.sum((target == 1) | (guess > threshold)).item()
    if union == 0:
        return 1.0
    return intersection / union