import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Callable, List, Union
from few_shot.metrics import NAMED_METRICS


def evaluate(model: nn.Module, dataloader: DataLoader, metrics: List[Union[str, Callable]],
             prepare_batch: Callable, loss_func: Callable, prefix: str = 'val_', suffix: str=''):
    logs, seen = {}, 0
    totals = {m: 0 for m in metrics}
    
    if loss_func is not None:
        totals['loss'] = 0
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            x, y = prepare_batch(batch)
            y_pred = model(x)
            seen += x.shape[0]
            
            if loss_func is not None:
                totals['loss'] += loss_func(y_pred, y).item() * x.shape[0]
            
            for m in metrics:
                if isinstance(m, str):
                    v = NAMED_METRICS[m](y, y_pred)
                else:
                    v = m(y, y_pred)
                
                totals[m] += v * x.shape[0]
                
    for m in ['loss'] + metrics:
        logs[prefix + m + suffix] = totals[m] / seen # type: ignore
        
    return logs