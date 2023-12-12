import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

CHECKPOINT_PATH = 'saved_model'

def train_model(model_class, train_loader: DataLoader, val_loader: DataLoader, proto_dim=64, learning_rate=2e-4):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, model_class.__name__),
                         accelerator='gpu', devices=1, max_epochs=1, enable_progress_bar=True,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc'),
                                    LearningRateMonitor('epoch')])
    
    # check pretrained model
    pretrained_filename = os.path.join(CHECKPOINT_PATH, model_class.__name__ + '.ckpt')
    if os.path.isfile(pretrained_filename):
        print(f'Found pretrained model {model_class.__name__}, loading .................')
        model = model_class.load_from_checkpoint(pretrained_filename)
    else:
        model = model_class(proto_dim, learning_rate)
        trainer.fit(model, train_loader, val_loader)
        model = model_class.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        
    return model