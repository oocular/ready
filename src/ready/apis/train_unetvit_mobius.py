import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms

from ready.models.unetvit import UNetViT
from ready.utils.datasets import MOBIOUSDataset_unetvit
from ready.utils.utils import DeviceDataLoader, get_default_device, precision, recall

from ready.utils.utils import (DATASET_PATH, MODELS_PATH, DeviceDataLoader,
                               get_default_device, precision, recall)

import yaml
#python -m src.ready.apis.train_unetvit_mobius

with open(str(Path().absolute())+"/tests/config_test.yml", "r") as file:
    config_yaml = yaml.load(file, Loader=yaml.FullLoader)
DATASET_PATH = os.path.join(str(Path.home()), config_yaml['ABS_DATA_PATH'])
MODELS_PATH = DATASET_PATH + "/models" #TODO add an absolute model path

os.makedirs(MODELS_PATH, exist_ok=True)

print(f"****** Training started at: {datetime.now()} ******")

# -------------------------
# Define Transforms
# -------------------------
# For UNetViT, we resize to (512, 512) then apply color jitter and gaussian blur
resize = transforms.Resize((512, 512))
color_shift = transforms.ColorJitter(0.1, 0.1, 0.1, 0.1)
blurriness = transforms.GaussianBlur(3, sigma=(0.1, 2.0))
t = transforms.Compose([resize, color_shift, blurriness])
 
# -------------------------
# Load Dataset and Create Dataloaders
# -------------------------
dataset = MOBIOUSDataset_unetvit(DATASET_PATH, training=True, transform=t)

# Split into training and testing datasets (e.g., 90% train, 10% test)
test_ratio = 0.1
test_size = int(test_ratio * len(dataset))
train_size = len(dataset) - test_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(101)
)

BATCH_SIZE = 4
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# Move data to the appropriate device
device = get_default_device()
train_dataloader = DeviceDataLoader(train_dataloader, device)
test_dataloader = DeviceDataLoader(test_dataloader, device)

# -------------------------
# Initialize Model, Loss, Optimizer, Scheduler
# -------------------------
# UNetViT with 3-channel images and 6 classes (adjust n_classes if needed)
model = UNetViT(n_channels=3, n_classes=6, bilinear=True).to(device)
criterion = nn.CrossEntropyLoss()  # Alternatively, you can use FocalLoss if desired
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

# -------------------------
# Training Loop
# -------------------------
N_EPOCHS = 5  # Change as needed
best_val_loss = float("inf")
start_time = time.time()

for epoch in range(N_EPOCHS):
    # --- Training phase ---
    model.train()
    train_loss = 0.0
    for batch_idx, (images, masks) in enumerate(train_dataloader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.long())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        sys.stdout.write(
            f"\r[Epoch {epoch+1}/{N_EPOCHS}] "
            f"[Batch {batch_idx+1}/{len(train_dataloader)}] "
            f"[Loss: {loss.item():.4f}]"
        )
        sys.stdout.flush()

    avg_train_loss = train_loss / len(train_dataloader)

    # --- Validation phase ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in test_dataloader:
            outputs = model(images)
            loss = criterion(outputs, masks.long())
            val_loss += loss.item()

    avg_val_loss = val_loss / len(test_dataloader)
    print(f"\nEpoch {epoch+1} - Train Loss: {avg_train_loss:.5f}, Val Loss: {avg_val_loss:.5f}")

    # Save the best model based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model_save_path = os.path.join(
            MODELS_PATH, f"unetvit_epochs_{epoch+1}_valloss_{avg_val_loss:.5f}.pth"
        )
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved best model to: {model_save_path}")

    # Step the scheduler
    lr_scheduler.step()

# -------------------------
# Final Metrics and Completion
# -------------------------
elapsed = time.time() - start_time
print(f"\nTraining completed in {elapsed/60:.2f} minutes")
print(f"****** Training ended at: {datetime.now()} ******")
