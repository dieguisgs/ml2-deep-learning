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


class DCGAN_torch:
    """
    """

    def __init__(self, dataset: str = "mnist", data_dir: str = "data"):
        """
        """
        self._dataset=dataset

        self._train_dataset = self._load_dataset(dataset, data_dir)

        self._latent_dim = 100

        # Build and compile the discriminator
        self._discriminator = self._build_discriminator()
        self._criterion = nn.CrossEntropyLoss()
        # self._criterion = nn.BCELoss()
        self._optimizer = optim.Adam(self._discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # Build the generator
        self._generator = self._build_generator()

        # The generator takes an array of random elements (noise) as input and generates images
        # noise = torch.nn.Parameter(torch.randn((self._latent_dim, )))
        noise = torch.randn((128, self._latent_dim))
        # noise = noise.unsqueeze(0)
        self._generator.eval()  # set the model to evaluation mode
        images = self._generator(noise)

        # For the combined model we will only train the generator
        for param in self._discriminator.parameters():
            param.requires_grad = False
        # self._discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        # self._discriminator.eval()
        # validity = self._discriminator(images)

        # The combined model (stacked generator and discriminator)
        # trains the generator to fool the discriminator
        # self._combined = nn.Sequential(
        #     noise, 
        #     self._generator, 
        #     self._discriminator
        #     )
        self._combined = nn.Sequential(self._generator, self._discriminator)
        # self._combined.eval()  # set the combined model to evaluation mode
        
        # Give a name to each of the layers of the combined model
        # self._combined._name[0] = 'Noise'
        # self._combined._name[1] = 'Generator'
        # self._combined._name[2] = 'Discriminator'

        # print('\n')
        self._combined._name = 'Generator_Discriminator'
        # Summary
        # print(self._combined)
        # summary(self._combined, input_size=(self._latent_dim, 1, 1))


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

        """
        trainloader = torch.utils.data.DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True)
        train_iter = iter(trainloader)

        # Adversarial loss ground truths
        valid = torch.ones(batch_size, dtype=int)
        fake = torch.zeros(batch_size, dtype=int)

        print("\n\nTraining...")

        # Create progress bar
        pbar = trange(1, epochs + 1, total=epochs, unit="epoch", file=sys.stdout)

        # Training loop
        for epoch in tqdm(range(epochs)):

            # ---------------------
            #  TRAIN DISCRIMINATOR
            # ---------------------

            # Select a random batch of images
            try:
                imgs, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(trainloader)
            # idx = np.random.randint(0, len(self._train_dataset), batch_size)
            # print(self._train_dataset)
            # imgs = self._train_dataset[idx]

            # Sample noise as generator input
            noise = torch.randn((batch_size, self._latent_dim))

            # Generate a batch of new images
            # self._generator.eval()
            # self._discriminator.train()
            gen_imgs = self._generator(noise)

            # Zero your gradients for every batch!
            self._optimizer.zero_grad()

            # Compute the discriminator loss
            d_loss_real = self._criterion(self._discriminator(imgs), valid)
            d_loss_fake = self._criterion(self._discriminator(gen_imgs), fake)
            d_loss = 0.5 * (d_loss_real.mean() + d_loss_fake.mean())

            # Compute the discriminator accuracy
            d_acc_real = (self._discriminator(imgs).ge(0.5).float().mean().item())
            d_acc_fake = (self._discriminator(gen_imgs.detach()).lt(0.5).float().mean().item())
            d_acc = 0.5 * (d_acc_real + d_acc_fake)

            # d_loss = 0.5 * np.add(d_loss_real.detach().numpy(), d_loss_fake.detach().numpy())
            # print(d_loss)
            # d_loss = torch.from_numpy(np.asarray(d_loss))
            # print(d_loss)

            # Compute the loss and its gradients
            d_loss.backward()

            # Adjust learning weights
            self._optimizer.step()

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train generator by using the self._combined model

            # Sample noise as generator input
            # noise = torch.randn((batch_size, self._latent_dim))

            # # Generate a batch of new images
            # self._combined.train()

            # # Zero your gradients for every batch!
            # self._optimizer.zero_grad()

            # # Compute the loss and its gradients
            # g_loss = self._criterion(self._combined(noise), valid)
            # g_loss.backward()

            # # Adjust learning weights
            # self._optimizer.step()

            # Train the generator
            # self._combined.train()
            g_loss = self._criterion(self._combined(noise), valid)

            # Print progress
            # print(d_loss)
            # print(d_loss.item())
            pbar.set_description("[Discriminator loss: %.4f, accuracy: %.2f%%] - [Generator loss: %.4f]"
                                 % (d_loss.item(), 100 * d_acc, g_loss))
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
    def _load_dataset(dataset: str, data_dir: str):
        """
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

        # trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)

        return trainset
    

    def _build_discriminator(self):
        """
        """
        model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Flatten(),
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
        # summary(model, input_size=(1, 28, 28))

        # Return the model
        return model

    
    def _build_generator(self):
        """
        """
        
        model = nn.Sequential(
            nn.Linear(in_features=self._latent_dim, out_features=7 * 7 * 128),

            nn.BatchNorm1d(num_features=7 * 7 * 128, momentum=0.9),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Unflatten(dim=1, unflattened_size=(128, 7, 7)),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64, momentum=0.9),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=1, momentum=0.9),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

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
        # summary(model, input_size=(self._latent_dim, 1, 1))

        # Return the model
        return model
    

    def _generate_samples(self, epoch: int,  images_dir: str, figure_size: Tuple[int, int] = (10, 10)):
        """Saves a .png figure composed of several generated images arranged as specified in the figure_size parameter.

        Args:
            epoch: Training epoch number. Used to name the saved image.
            figure_size: (rows, columns) layout of the output image.

        """
        # Create folder if it does not exist
        os.makedirs(images_dir, exist_ok=True)
        
        # Generate images
        rows, cols = figure_size
        noise = torch.randn((rows*cols, self._latent_dim))
        self._generator.eval()
        generated_images = self._generator(noise)
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