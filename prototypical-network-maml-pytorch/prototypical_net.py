from typing import Tuple
import torch
import torch.nn as nn
import torchvision
import torch.functional as F
import pytorch_lightning as pl


def get_convnet(output_size):
    """
     Create a convolutional neural network. It is used to train and test the neural network.
     
     Args:
     	 output_size: Number of classes in the output.
     
     Returns: 
     	 : class : ` torchvision. models. DenseNet `
    """
    return torchvision.models.DenseNet(growth_rate=32, block_config=(6, 6, 6, 6), bn_size=2,
                                        num_init_features=64, num_classes=output_size)
    
def split_batch(imgs, targets):
    support_imgs, query_imgs = imgs.chunk(2, dim=0)  # split into support set and query set
    support_targets, query_targets = targets.chunk(2, dim=0)
    return support_imgs, query_imgs, support_targets, query_targets
    
class PrototypicalNet(pl.LightningModule):
    def __init__(self, proto_dim, learning_rate):
        super(PrototypicalNet, self).__init__()
        self.save_hyperparameters()
        self.proto_dim, self.learning_rate = proto_dim, learning_rate
        self.model = get_convnet(output_size=proto_dim)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[140, 180], gamma=0.1)
        return [optimizer], [scheduler]
    
    @staticmethod
    def calculate_prototypes(features: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates class prototypes from features and targets.
        Args:
            features: torch.Tensor - encoded raw data to L-dimension vector by embedding function f_theta
            targets: torch.Tensor - labels of coresponding features
        Returns: 
            prototypes: Calculated prototypes
            classes: Available classes
        """
        classes, _ = torch.unique(targets).sort()  # determine available classes
        prototypes = []
        for c in classes:
            prototypes.append(features[torch.where(targets==c)[0]].mean(dim=0))  # average class feature vectors
        prototypes = torch.stack(prototypes, dim=0)
        return prototypes, classes
    
    def classify_features(self, prototypes: torch.Tensor, classes: torch.Tensor, features: torch.Tensor, 
                          targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Classify features based on prototypes and calculate accuracy.
        Args:
            prototypes: torch.Tensor - Prototypes for each class
            classes: torch.Tensor - Class labels
            features: torch.Tensor - Features to classify
            targets: torch.Tensor - True class labels
        Returns: 
            preds: torch.Tensor - Predicted class probabilities
            labels: torch.Tensor - True class labels
            acc: torch.Tensor - Classification accuracy
        - Calculate Euclidian distance between features and prototypes
        - Convert distances to probabilities using softmax
        - Compare predicted and true class labels to calculate accuracy, convert to one-hot encoder and argmax index
        - Return predictions, true labels and accuracy
        """
        distance = torch.pow(prototypes[None, :] - features[:, None], 2).sum(dim=2)  # calculate euclidian distance
        preds = torch.nn.functional.log_softmax(-distance, dim=1)  # convert from distance (negative) to probability
        labels = (classes[None, :] == targets[:, None]).long().argmax(dim=-1) 
        acc = (preds.argmax(dim=1) == labels).float().mean()
        return preds, labels, acc
    
    def calculate_loss(self, batch, mode):
        """
        Calculates loss for a batch during training or validation.
        Args:
            batch: Batch of images and targets
            mode: Whether this is for 'train' or 'val' mode
        Returns: 
            loss: Cross entropy loss between predictions and targets
        """
        imgs, targets = batch
        features = self.model(imgs)  # encode all images of support and query set
        support_features, query_features, support_targets, query_targets = split_batch(features, targets)  
        prototypes, classes = PrototypicalNet.calculate_prototypes(support_features, support_targets)
        predictions, labels, acc = self.classify_features(prototypes, classes, query_features, query_targets)
        loss = torch.nn.functional.cross_entropy(predictions, labels)
        
        self.log(f'{mode}_loss', loss); self.log(f'{mode}_acc', acc)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.calculate_loss(batch, mode='train')
    
    def validation_step(self, batch, batch_idx):
        _ = self.calculate_loss(batch, mode='val')
        
        
     