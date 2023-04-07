# Python base libraries
import os
import sys
import time
from typing import Tuple

# Data science libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Other libraries
from tqdm.notebook import tqdm
from tqdm import trange

# Deep learning libraries
import torch
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchsummary import summary

# Set the device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DCGAN_torch:
    """
    Deep Convolutional Generative Adversarial Network (DCGAN) in PyTorch.

    Args:
    -------
    dataset: str
        Name of the dataset to use. Default: "mnist".
    data_dir: str
        Directory where the dataset is stored. Default: "data".

    Examples:
    ---------
    >>> from src.dcgan_torch import DCGAN_torch
    >>> dcgan = DCGAN_torch(dataset='mnist', data_dir='data')
    >>> dcgan.train(epochs=100, batch_size=32, save_interval=100, images_dir='images/pytorch_mnist', models_dir='models')
    """

    def __init__(self, dataset: str = "mnist", data_dir: str = "data"):
        """
        Initialize the DCGAN_torch class.

        Args:
        -------
        dataset: str
            Name of the dataset to use. Default: "mnist".
        data_dir: str
            Directory where the dataset is stored. Default: "data".
        """
        # Define the dataset
        self._dataset=dataset
        # Load the dataset
        self._train_dataset = self._load_dataset(dataset, data_dir)
        # Define the number of variables in the latent space
        self._latent_dim = 100

        # ===========================================================
        # BUILD DISCRIMINATOR
        # ===========================================================
        self._discriminator = self._build_discriminator()
        self._loss = nn.BCEWithLogitsLoss()
        self._discriminator_optimizer = optim.Adam(self._discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # ===========================================================
        # BUILD GENERATOR
        # ===========================================================
        self._generator = self._build_generator()
        self._generator_optimizer = optim.Adam(self._generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Set both models to training mode
        self._generator.train()
        self._discriminator.train()


    def train(self, epochs: int = 1000, batch_size: int = 128, save_interval: int = 50, images_dir: str = "images/mnist", models_dir: str = "models"):
        """Train the discriminator and the generator.

        Saves both models upon completion.

        Args:
        -------
        epochs: 
            Number of times the dataset is passed forward and backward through the neural network.
        batch_size: 
            Number of examples used in one iteration.
        save_interval: 
            Save interval for sample generated images [epochs]. Set to 0 to disable.
        images_dir:
            Directory where the generated images will be saved.
        models_dir:
            Directory where the trained models will be saved.
        """
        trainloader = torch.utils.data.DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True)
        # train_iter = iter(trainloader)

        # Adversarial loss ground truths
        # valid = torch.ones(batch_size, dtype=int)
        # fake = torch.zeros(batch_size, dtype=int)

        print("\n\nTraining...")

        # Create progress bar
        pbar = trange(1, epochs + 1, total=epochs, unit="epoch", file=sys.stdout)

        # Training loop
        # Epochs
        for epoch in tqdm(range(epochs)):
            d_total_loss = 0
            g_total_loss = 0
            # gen_out = 0
            # Batches
            for batch, labels in trainloader:
                # ==================================
                #  TRAIN DISCRIMINATOR
                # ==================================
                # REAL IMAGES
                # Select a random batch of images
                # Set discriminator gradients to zero in every batch
                self._discriminator_optimizer.zero_grad()
                batch = batch.to(device)
                # Get discriminator output for real images
                d_real_output = self._discriminator(batch.float())

                # FAKE IMAGES
                noise = torch.rand(batch_size, self._latent_dim)*2 - 1
                noise = noise.to(device)
                # Use detach to avoid training the generator
                fake_images = self._generator(noise.float().unsqueeze(2).unsqueeze(3)).detach()
                d_fake_output = self._discriminator(fake_images.float())

                # DISCRIMINATOR LOSS
                d_loss = self._discriminator_loss(d_real_out=d_real_output, d_fake_out=d_fake_output)
                d_total_loss += d_loss

                # Backpropagate discriminator loss
                d_loss.backward()
                # Update discriminator weights
                self._discriminator_optimizer.step()

                # ==================================
                #  TRAIN GENERATOR
                # ==================================
                # Set generator gradients to zero in every batch
                self._generator_optimizer.zero_grad()

                # Generate fake images
                g_output = self._generator(noise.float().unsqueeze(2).unsqueeze(3))
                # Get discriminator output for fake images
                combined_output = self._discriminator(g_output.float())

                g_loss = self._generator_loss(combined_model_out=combined_output)
                g_total_loss += g_loss

                # Backpropagate generator loss
                g_loss.backward()
                # Update generator weights
                self._generator_optimizer.step()

            pbar.set_description("[Discriminator loss: %.4f] - [Generator loss: %.4f]"
                                 % (d_total_loss.item(), g_total_loss.item()))
            pbar.update(1)
            time.sleep(0.01)  # Prevents a race condition between tqdm and print statements.

            # If at save interval => save generated image samples
            if save_interval > 0 and epoch % save_interval == 0:
                self._generate_samples(epoch, images_dir)

        # Save final models
        if not os.path.isdir('models'):
            os.makedirs('models', exist_ok=True)

        # Save both models
        torch.save(self._generator, os.path.join(models_dir, f'pytorch_{self._dataset}_generator.h5'))
        torch.save(self._discriminator, os.path.join(models_dir, f'pytorch_{self._dataset}_discriminator.h5'))


    @staticmethod
    def _load_dataset(dataset: str, data_dir: str) -> torch.utils.data.Dataset:
        """
        Loads the dataset. Currently supports MNIST and Fashion MNIST.

        Args:
        -------
        dataset: str
            Name of the dataset to load (mnist or fashion_mnist).
        data_dir: str
            Directory where to save the dataset.

        Returns:
        -------
        trainset: torch.utils.data.Dataset
            Training dataset.
        """
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        # Create folder directories where to save the dataset
        os.makedirs(data_dir, exist_ok=True)

        # Load the dataset
        if dataset == "mnist":
            trainset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
            # testset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        elif dataset == "fashion_mnist":
            trainset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
            # testset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
        else:
            raise ValueError("Invalid dataset name")

        return trainset
    

    def _build_discriminator(self) -> torch.nn.Sequential:
        """
        Builds the discriminator model.

        Returns:
        -------
        model: torch.nn.Sequential
            Discriminator model.
        """
        model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),

            nn.Sigmoid()
            )

        # Initialize weights of all the layers
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.normal_(module.weight.data, mean=0.0, std=0.02)

        print('\n')
        model._name = 'Discriminator'

        # Summary of the model
        print(model)

        # Return the model
        return model

    
    def _build_generator(self) -> torch.nn.Sequential:
        """
        Builds the generator model.

        Returns:
        -------
        model: torch.nn.Sequential
            Generator model.
        """
        model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self._latent_dim, out_channels=128, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),

            nn.Tanh()
        )


        # Initialize weights of all the layers
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.normal_(module.weight.data, mean=0.0, std=0.02)

        print('\n')
        model._name = 'Generator'
        
        # Summary of the model
        print(model)

        # Return the model
        return model
    

    def _generate_samples(self, epoch: int,  images_dir: str, figure_size: Tuple[int, int] = (10, 10)):
        """
        Saves a .png figure composed of several generated images arranged as specified in the figure_size parameter.

        Args:
        -------
        epoch: int
            Training epoch number. Used to name the saved image.
        images_dir: str
            Directory where to save the generated images.
        figure_size: Tuple[int, int]
            Size of the figure to save. The first element of the tuple is the number of rows, the second element is the number of columns.
        """
        # Create folder if it does not exist
        os.makedirs(images_dir, exist_ok=True)
        
        # Generate images
        rows, cols = figure_size
        noise = torch.randn((rows*cols, self._latent_dim))
        self._generator.eval()
        generated_images = self._generator(noise.float().unsqueeze(2).unsqueeze(3)).detach()
        generated_images = 0.5 * generated_images + 0.5  # Rescale images to [0, 1]

        # Build and save a single figure with all the generated images side-by-side
        plt.figure(figsize=figure_size)

        for i in range(generated_images.shape[0]):
            plt.subplot(rows, cols, i+1)
            plt.imshow(generated_images.detach().numpy()[i, 0, :, :], interpolation='nearest', cmap='gray_r')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, "epoch_%04d.png" % epoch))
        plt.close()


    def _discriminator_loss(self, d_real_out: torch.Tensor, d_fake_out: torch.Tensor) -> torch.Tensor:
        """
        Computes the discriminator loss.

        Args:
        -------
        d_real_out: torch.Tensor
            Output of the discriminator for real images.
        d_fake_out: torch.Tensor
            Output of the discriminator for fake images.

        Returns:
        -------
        total_loss: torch.Tensor
            Total discriminator loss.
        """
        # DISCRIMINATOR LOSS FOR REAL IMAGES
        real_label = torch.ones(d_real_out.size()[0], 1).to(device)
        real_loss = self._loss(d_real_out.squeeze(), real_label.squeeze())

        # DISCRIMINATOR LOSS FOR FAKE IMAGES
        fake_label = torch.zeros(d_fake_out.size()[0], 1).to(device)
        fake_loss = self._loss(d_fake_out.squeeze(), fake_label.squeeze())

        # TOTAL DISCRIMINATOR LOSS
        total_loss = (real_loss + fake_loss)
        return total_loss
    
    
    def _generator_loss(self, combined_model_out: torch.Tensor) -> torch.Tensor:
        """
        Computes the generator loss.

        Args:
        -------
        combined_model_out: torch.Tensor
            Output of the combined model (generator + discriminator).
        
        Returns:
        -------
        gen_loss: torch.Tensor
            Generator loss.
        """
        label = torch.ones(combined_model_out.size()[0], 1).to(device)
        gen_loss = self._loss(combined_model_out.squeeze(), label.squeeze())
        return gen_loss