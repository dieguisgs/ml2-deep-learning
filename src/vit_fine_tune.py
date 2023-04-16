# Imports
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    CenterCrop, 
    Compose, 
    Normalize, 
    RandomHorizontalFlip,
    RandomResizedCrop, 
    Resize, 
    ToTensor
)
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification, 
    AdamW
)

"""
Below is the code for the ViT model and the training, validation, and testing steps.

We start by defining the constants for the data directories, hyperparameters, and the processor.
"""

# Data directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'src', 'dataset')
TRAIN_DIR = os.path.join(DATA_DIR, 'training')
VAL_DIR = os.path.join(DATA_DIR, 'validation')

# Hyperparameters
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_TEST = 4
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
NUM_CLASSES = len(os.listdir(TRAIN_DIR))

# Processor
PROCESSOR = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

# Processor parameters
IMG_SIZE = PROCESSOR.size
IMG_MEAN = PROCESSOR.image_mean
IMG_STD = PROCESSOR.image_std

class ViTLightningModule(pl.LightningModule):
    """
    Class to define the ViT model and the training, validation and testing steps

    This code defines a ViT (Vision Transformer) model and its training, validation, and testing steps using PyTorch Lightning. 
    PyTorch Lightning is a high-level interface for PyTorch that makes it easier to train models and organize the code.

    The code first initializes the model and defines the transforms for the train, validation, and test sets. 
    It then loads the data and creates data loaders for each set.

    Next, the forward pass is defined as simply calling the forward method of the ViT model. 
    The training, validation, and testing steps are also defined using the  training_step, validation_step, and test_step methods respectively. 
    These methods take a batch of data and compute the loss and metrics for that batch.

    Finally, the optimizer is defined using the AdamW algorithm and the data loaders are defined using the train_dataloader, val_dataloader, and test_dataloader methods.

    Attributes:
    ----------
    train_transforms : Compose
        The transforms to apply to the train set
    val_transforms : Compose
        The transforms to apply to the validation set
    test_transforms : Compose
        The transforms to apply to the test set
    train_data : ImageFolder
        The train set
    val_data : ImageFolder
        The validation set
    test_data : ImageFolder
        The test set
    train_loader : DataLoader
        The data loader for the train set
    val_loader : DataLoader
        The data loader for the validation set
    test_loader : DataLoader
        The data loader for the test set
    id2label : dict
        The dictionary mapping the class id to the class label
    label2id : dict
        The dictionary mapping the class label to the class id
    vit : ViTForImageClassification
        The ViT model
    
    Methods:
    -------
    forward(x)
        Defines the forward pass
    training_step(batch, batch_idx)
        Defines the training step
    validation_step(batch, batch_idx)
        Defines the validation step
    test_step(batch, batch_idx)
        Defines the test step
    configure_optimizers()
        Defines the optimizer
    train_dataloader()
        Defines the data loader for the train set
    val_dataloader()
        Defines the data loader for the validation set
    test_dataloader()
        Defines the data loader for the test set
    """
    def __init__(self):
        """
        This method initializes the ViT model and defines the transforms for the train, validation, and test sets.

        Parameters:
        ----------
        None

        Returns:
        -------
        None
        """
        # Call the parent constructor
        super(ViTLightningModule, self).__init__()

        # Define the transforms for the train, validation and test sets
        self.train_transforms = Compose([
            # Resize the image to the size expected by the ViT model randomly
            RandomResizedCrop(IMG_SIZE['height']),
            # Randomly flip the image horizontally
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])

        self.val_transforms = Compose([
            Resize(IMG_SIZE['height']),
            CenterCrop(IMG_SIZE['height']),
            ToTensor(),
            Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])

        self.test_transforms = Compose([
            Resize(IMG_SIZE['height']),
            CenterCrop(IMG_SIZE['height']),
            ToTensor(),
            Normalize(mean=IMG_MEAN, std=IMG_STD)
        ])


        # Define the data
        self.train_data = ImageFolder(
            TRAIN_DIR,
            transform=self.train_transforms
        )

        self.val_data = ImageFolder(
            VAL_DIR,
            transform=self.val_transforms
        )

        self.test_data = ImageFolder(
            VAL_DIR,
            transform=self.test_transforms
        )

        # Define the data loaders
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=BATCH_SIZE_TRAIN,
            shuffle=True,
            num_workers=NUM_WORKERS
        )

        self.val_loader = DataLoader(
            self.val_data,
            batch_size=BATCH_SIZE_TEST,
            shuffle=False,
            num_workers=NUM_WORKERS
        )

        self.test_loader = DataLoader(
            self.test_data,
            batch_size=BATCH_SIZE_TEST,
            shuffle=False,
            num_workers=NUM_WORKERS
        )

        # Define the label to id and id to label dictionaries
        self.id2label = {v: k for k, v in self.train_data.class_to_idx.items()}
        self.label2id = {label:id for id, label in self.id2label.items()}

        # Define the model
        self.vit = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            num_labels=NUM_CLASSES,
            id2label=self.id2label,
            label2id=self.label2id
        )
    

    def forward(self, x):
        """
        Defines the forward pass

        Parameters:
        ----------
        self: ViTLightningModule
            The object which calls the method
        x : Tensor
            The input tensor
        
        Returns:
        -------
        Tensor
            The output tensor
        """
        return self.vit(x)
    
    # Define the training, validation and testing steps
    def training_step(self, batch, batch_idx):
        """
        Defines the training step

        Parameters:
        ----------
        self: ViTLightningModule
            The object which calls the method
        batch : Tensor
            The input tensor
        batch_idx : int
            The index of the batch

        Returns:
        -------
        None
            Logs the training loss
        """
        x, y = batch
        loss = self.vit(x, labels=y).loss
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step

        Parameters:
        ----------
        self: ViTLightningModule
            The object which calls the method
        batch : Tensor
            The input tensor
        batch_idx : int
            The index of the batch
        
        Returns:
        -------
        None
            Logs the validation loss
        """
        x, y = batch
        loss = self.vit(x, labels=y).loss
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        """
        Defines the test step

        Parameters:
        ----------
        self: ViTLightningModule
            The object which calls the method
        batch : Tensor
            The input tensor
        batch_idx : int
            The index of the batch
        
        Returns:
        -------
        None
            Logs the test loss
        """
        x, y = batch
        loss = self.vit(x, labels=y).loss
        self.log("test_loss", loss)
    
    # Define the optimizer
    def configure_optimizers(self):
        """
        Defines the optimizer

        Parameters:
        ----------
        self: ViTLightningModule
            The object which calls the method
        
        Returns:
        -------
        AdamW
            The AdamW optimizer
        """
        return AdamW(
            self.vit.parameters(),
            lr=LEARNING_RATE
        )
    
    # Define the data loaders    
    def train_dataloader(self):
        """
        Defines the data loader for the train set

        Parameters:
        ----------
        self: ViTLightningModule
            The object which calls the method
        
        Returns:
        -------
        DataLoader
            The data loader for the train set
        """
        return self.train_loader
    
    def val_dataloader(self):
        """
        Defines the data loader for the validation set

        Parameters:
        ----------
        self: ViTLightningModule
            The object which calls the method
        
        Returns:
        -------
        DataLoader
            The data loader for the validation set
        """
        return self.val_loader
    
    def test_dataloader(self):
        """
        Defines the data loader for the test set

        Parameters:
        ----------
        self: ViTLightningModule
            The object which calls the method

        Returns:
        -------
        DataLoader
            The data loader for the test set
        """
        return self.test_loader