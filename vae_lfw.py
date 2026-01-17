import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

import os
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matplotlib.pyplot as plt

DATA_DIR = "./lfw-deepfunneled" # root/lfw-deepfunneled
MODEL_PATH = "vae_lfw.pth"
IMAGE_SIZE = 64
LATENT_DIM = 6
BATCH_SIZE = 128
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# to test that root directory is correct

# print(f"Loaded {len(dataset)} images from {DATA_DIR}")

# this code to check 25 samples from the dataset

# dataiter = iter(dataloader)
# image = next(dataiter)

# num_samples = 25
# sample_images = [image[0][i,0] for i in range(num_samples)] 

# fig = plt.figure(figsize=(5, 5))
# grid = ImageGrid(fig, 111, nrows_ncols=(5, 5), axes_pad=0.1)

# for ax, im in zip(grid, sample_images):
#     ax.imshow(im, cmap='gray')
#     ax.axis('off')

# plt.show()

class VAE(nn.Module):
    
    def __init__(self, latent_dim):
        super(VAE,self).__init__()
        self.latent_dim = latent_dim
        
        # the encoder layers

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 3x64x64 → 32x32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x32x32 → 64x16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 64x16x16 → 128x8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# 128x8x8 → 256x4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.mean_layer = nn.Linear(256 * 4 * 4, latent_dim)
        self.logvar_layer = nn.Linear(256 * 4 * 4, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)

        # the decoder layers

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 256x4x4 → 128x8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 128x8x8 → 64x16x16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 64x16x16 → 32x32x32
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # 32x32x32 → 3x64x64
            nn.Tanh()  # output ∈ [-1, 1]
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 256, 4, 4) # changing encoder input to go into Conv2D layer
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.mse_loss(x, x_hat, reduction='sum') # mse loss better used for 3x64x64 images like in the LFW dataset
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD

# need to train the model iff the model file doesnt exist

def train(model, optimizer, epochs, device):
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded from disk.")
    else:
        model.train()
        for epoch in range(epochs):
            for batch_idx, (imgs, _) in enumerate(dataloader):
                imgs = imgs.to(device)
                optimizer.zero_grad()
                x_hat, mean, logvar = model(imgs)
                loss = loss_function(imgs, x_hat, mean, logvar)
                loss.backward()
                optimizer.step()
        
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

class VAE_GUI:
    def __init__(self, model, latent_dim=6, device=DEVICE):
        self.model = model
        self.model.eval()
        self.latent_dim = latent_dim
        self.device = device

        # --- Main Window Setup ---
        self.root = tk.Tk()
        self.root.title("6 Dimensional Latent Space VAE")
        self.root.geometry("1000x600")  # wider layout

        # --- Main layout frames ---
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left: Latent Variables panel
        left_frame = ttk.Frame(main_frame, padding=20)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Right: Reconstructed image panel
        right_frame = ttk.Frame(main_frame, padding=20)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Left section: Sliders ---
        title_label = ttk.Label(left_frame, text="Latent Variables", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)

        self.sliders = []
        self.latent_vars = np.zeros(self.latent_dim)

        for i in range(self.latent_dim):
            frame = ttk.Frame(left_frame)
            frame.pack(pady=5)
            label = ttk.Label(frame, text=f"z{i+1}", width=5)
            label.pack(side=tk.LEFT)
            slider = ttk.Scale(frame, from_=-3, to=3, orient='horizontal', length=200,
                               command=lambda val, idx=i: self.update_latent(idx, val))
            slider.pack(side=tk.RIGHT, padx=10)
            self.sliders.append(slider)

        # --- Right section: Image display ---
        image_label = ttk.Label(right_frame, text="Reconstructed Image", font=("Arial", 14, "bold"))
        image_label.pack(pady=10)

        image_box = ttk.Frame(right_frame, borderwidth=2, relief="groove", padding=10)
        image_box.pack(pady=10, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.ax.axis("off")

        self.canvas = FigureCanvasTkAgg(self.fig, master=image_box)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Initial display
        self.update_image()

    def update_latent(self, idx, val):
        self.latent_vars[idx] = float(val)
        self.update_image()

    def update_image(self):
        z = torch.tensor(self.latent_vars, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            recon = self.model.decode(z).cpu()
        img = recon[0]
        img = (img * 0.5 + 0.5).clamp(0, 1)  # unnormalize to [0,1]
        npimg = np.transpose(img.numpy(), (1, 2, 0))
        self.ax.clear()
        self.ax.imshow(npimg)
        self.ax.axis("off")
        self.canvas.draw()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    model = VAE(LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model, optimizer, EPOCHS, DEVICE)
    gui = VAE_GUI(model)
    gui.run()
