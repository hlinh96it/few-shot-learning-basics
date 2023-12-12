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
    """
     Split images and targets into support and query. This is used to reduce the number of queries to one batch at a time
     
     Args:
     	 imgs: A list of images in shape ( batch_size num_images )
     	 targets: A list of targets in shape ( batch_size num_targets )
     
     Returns: 
     	 A tuple of support images and query images for each
    """
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
    def calculate_prototypes(features, targets):
        """
        Calculate prototypes for each class. This is a helper function to make it easier to use in inference
        
        Args:
            features: torch. Tensor of shape [ batch_size, num_features ]
            targets: torch. Tensor of shape [ batch_size, num_classes ]
        
        Returns: 
            tuple of prototypes and classes (sorted by class)
        """
        classes, _ = torch.unique(targets).sort()  # determine available classes
        prototypes = []
        for c in classes:
            prototypes.append(features[torch.where(targets==c)[0]].mean(dim=0))  # average class feature vectors
        prototypes = torch.stack(prototypes, dim=0)
        return prototypes, classes
    
    def classify_features(self, prototypes, classes, features, targets):
        """
        Classify features according to prototypes. This is a wrapper around torch.nn.log_softmax 
        that takes into account the euclidian distance between the features and prototypes
        
        Args:
            prototypes: Tensor of shape [ batch_size n_features ]
            classes: Tensor of shape [ batch_size n_classes ]
            features: Tensor of shape [ batch_size n_features ]
            targets: Tensor of shape [ batch_size ] with target values
        
        Returns: 
            Tuple of ( logits labels accuracy ) for each class
        """
        distance = torch.pow(prototypes[None, :] - features[:, None], 2).sum(dim=2)  # calculate euclidian distance
        preds = torch.nn.functional.log_softmax(-distance, dim=1)
        labels = (classes[None, :] == targets[:, None]).long().argmax(dim=-1)
        acc = (preds.argmax(dim=1) == labels).float().mean()
        return preds, labels, acc
    
    def calculate_loss(self, batch, mode):
        """
        Calculate loss for a batch of images
        
        Args:
            batch: tuple of images and targets. Each image is a 2 - dim numpy array of shape 
        [ batch_size height width 3 ]
            mode: 'train' or 'test '. If 'train' the batch is split into support and query sets.
        
        Returns: 
            floating - point loss in range [ 0 1 ]
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
        
        
     