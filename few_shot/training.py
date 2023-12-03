import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from few_shot.callbacks import DefaultCallback, ProgressBarLogger, CallbackList, Callback

from typing import List, Callable, Union
from metrics import NAMED_METRICS


def gradient_step(model: nn.Module, optimiser: Optimizer, loss_fn: Callable, 
                  x: torch.Tensor, y: torch.Tensor, **kwargs):
    """Takes a single gradient step.

    Arguments
        model: Model to be fitted
        optimiser: Optimiser to calculate gradient step from loss
        loss_fn: Loss function to calculate between predictions and outputs
        x: Input samples
        y: Input targets
    """
    model.train()
    optimiser.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimiser.step()

    return loss, y_pred


def batch_metrics(model: nn.Module, y_pred: torch.Tensor, y: torch.Tensor,
                  metrics: List[Union[str, Callable]], batch_logs: dict):
    """Calculate metrics for current batch

    Args:
        model (nn.Module): model being fitted
        y_pred (torch.Tensor): predictions for a particular batch
        y (torch.Tensor): labels for particular batch
        metrics (List[Union[str, Callable]]): name and function to calculate metrics
        batch_logs (dict): dict of logs for current batch
    """
    model.eval()
    for m in metrics:
        if isinstance(m, str):
            batch_logs[m] = NAMED_METRICS[m](y, y_pred)
        else:
            batch_logs = m(y, y_pred)
    return batch_logs


def fit_model(model: nn.Module, optimizer: Optimizer, loss_fn: Callable, epochs: int, dataloader: DataLoader, 
              prepare_batch: Callable, metrics: List[Union[str, Callable]], callbacks: List[Callback],
              verbose: bool=True, fit_function: Callable=gradient_step, fit_function_kwargs: dict={}):
    num_batches, batch_size = len(dataloader), dataloader.batch_size
    callbacks = CallbackList([DefaultCallback()] + (callbacks or []) + [ProgressBarLogger()])
    callbacks.set_model(model)
    callbacks.set_params({
        'num_batches': num_batches, 'batch_size': batch_size, 'verbose': verbose,
        'metrics': metrics, 'prepare_batch': prepare_batch, 'loss_fn': loss_fn, 'optimizer': optimizer
    })
    
    for epoch in range(1, epochs+1):
        callbacks.on_batch_begin(epoch)
        epoch_log = {}
        for batch_index, batch in enumerate(dataloader):
            callbacks.on_batch_begin(batch=batch_index, logs=dict(batch=batch_index, size=64))
            x, y = prepare_batch(batch)
            loss, y_pred = fit_function(model, optimizer, loss_fn, x, y, **fit_function_kwargs)
            
            batch_end_logs = batch_metrics(model, y_pred, y, metrics, batch_logs=dict(batch=batch_index, size=64))
            callbacks.on_batch_end(batch=batch_index, logs=batch_end_logs)
        
        callbacks.on_epoch_end(epoch=epoch, logs=epoch_log)
    callbacks.on_train_end()
            
    
    
