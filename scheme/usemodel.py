"""
Functions used to process data using the model.

"""

import numpy as np
import torch
from torch import nn
from sklearn import metrics


def get_dev_metrics(model, loader, dev_subset=None, annotator_indices=None):

    # Make inference
    target, output = inference(model, loader, dev_subset=dev_subset, annotator_indices=annotator_indices)
    # Compute loss
    criterion = nn.BCELoss()
    loss = criterion(torch.Tensor(output.reshape(-1)), torch.Tensor(target)).numpy().astype(float)
    # Compute accuracy
    acc = np.sum((output > 0.5) == target) / np.size(target, 0)

    return acc, loss


def train_batch(model, data, target, optimizer):
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.train()
    outputs = model(data.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))).reshape(-1)
    target = target.to(torch.float32)

    criterion = nn.BCELoss()
    loss = criterion(outputs, target.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    return


def evaluate(model, loader, bin_threshold=0.5, dev_subset=None, annotator_indices=None, return_pr=False):

    target, output = inference(model=model,
                               loader=loader,
                               bin_threshold=bin_threshold,
                               dev_subset=dev_subset,
                               annotator_indices=annotator_indices)
    performance = calc_metrics(target, output, return_pr=return_pr)

    return performance[:]


def calc_metrics(target, output, return_pr=False):
    auc = 100 * metrics.roc_auc_score(target, output)
    f1 = 100 * metrics.f1_score(target, np.round(output))

    if not return_pr:
        return auc, f1
    else:
        precision = 100 * metrics.precision_score(target, np.round(output))
        recall = 100 * metrics.recall_score(target, np.round(output))
        return auc, f1, precision, recall


def inference(model, loader, bin_threshold=0.5, dev_subset=None, annotator_indices=None):
    target, output = [], []

    if dev_subset is not None:
        generator = loader.generate(dev_subset=dev_subset,
                                    annotator_indices=annotator_indices,
                                    bin_threshold=bin_threshold)
    else:
        generator = loader.generate(annotator_indices=annotator_indices, bin_threshold=bin_threshold)

    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()
    for batch_x, batch_y in generator:
        target.append(batch_y)
        with torch.no_grad():
            prediction = model(batch_x.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
        output.append(prediction.cpu())

    target, output = np.concatenate(target, axis=0), np.concatenate(output, axis=0)
    target, output = target.reshape(-1), output.reshape(-1)

    return target, output
