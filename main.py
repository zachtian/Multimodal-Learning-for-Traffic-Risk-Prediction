import os
import logging
from typing import Tuple, List
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
from torchvision import transforms
from CollisionNet import CollisionNet
import matplotlib.pyplot as plt

# Constants
BATCH_SIZE = 64
NUM_EPOCHS = 300
LEARNING_RATE = 1e-4
TRAIN_DATA_DIR = "train_data"
TEST_DATA_DIR = "test_data"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class IoU(nn.Module):
    def __init__(self, smooth=1e-6):
        super(IoU, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets, percentile=90):
        # Apply sigmoid if the outputs are logits
        outputs = torch.sigmoid(outputs)

        # Flatten the tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # Compute thresholds using the specified percentile
        gt_threshold = torch.quantile(targets, percentile / 100.0)
        pred_threshold = torch.quantile(outputs, percentile / 100.0)

        # Apply thresholding to identify high-risk zones
        gt_high_risk = (targets >= gt_threshold).float()
        pred_high_risk = (outputs >= pred_threshold).float()

        # Calculate the intersection and union
        intersection = (gt_high_risk * pred_high_risk).sum()
        union = gt_high_risk.sum() + pred_high_risk.sum() - intersection

        # Compute IoU and return IoU loss
        iou = (intersection + self.smooth) / (union + self.smooth)
        return iou 
# Define the dataset
class CollisionDataset(Dataset):
    def __init__(self, root_dir: str, augment: bool = False, baseline = False):
        self.root_dir = root_dir
        self.augment = augment
        self.files = [f.split('_')[0] for f in os.listdir(root_dir) if f.endswith('_satellite.png')]
        self.transforms = self._get_transforms(augment)
        self.image_transforms = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.baseline = baseline

    def _get_transforms(self, augment: bool) -> transforms.Compose:
        if augment:
            return transforms.Compose([
                transforms.Resize((288, 288)),
                transforms.RandomCrop((256, 256)),
                transforms.RandomRotation(degrees=180),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((288, 288)),
                transforms.CenterCrop((256, 256)),
            ])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_id = self.files[idx]

        satellite = np.array(Image.open(os.path.join(self.root_dir, f"{file_id}_satellite.png"))).astype(np.float32) / 255.0
        satellite = torch.tensor(satellite).permute(2, 0, 1)
        satellite = self.image_transforms(satellite)

        building = torch.tensor(np.load(os.path.join(self.root_dir, f"{file_id}_building.npy")).astype(np.float32)).unsqueeze(0)
        traffic = torch.tensor(np.load(os.path.join(self.root_dir, f"{file_id}_traffic.npy")).astype(np.float32)).unsqueeze(0)
        heatmap = torch.tensor(np.load(os.path.join(self.root_dir, f"{file_id}_heatmap.npy")).astype(np.float32)).unsqueeze(0)

        if self.baseline:
            input_data = torch.cat([satellite, building], dim=0)
        else:    
            input_data = torch.cat([satellite, building, traffic], dim=0)
        combined = torch.cat([input_data, heatmap], dim=0)
        combined = self.transforms(combined)
        input_data, target = combined[:-1], combined[-1:]

        return input_data, target

# Define the Lightning module
class CollisionNetLightning(pl.LightningModule):
    def __init__(self, model: nn.Module, learning_rate: float = LEARNING_RATE):
        super(CollisionNetLightning, self).__init__()
        self.model = model
        self.IoU = IoU()
        self.learning_rate = learning_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.mse_loss(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        inputs, targets = batch
        outputs = self(inputs)
        # Compute metrics
        iou = self.IoU(outputs, targets)

        mse_loss = F.mse_loss(outputs, targets)
        rmse = torch.sqrt(mse_loss)

        # Log each metric separately
        self.log('val_iou', iou, prog_bar=True)
        self.log('val_rmse', rmse, prog_bar=True)

        # Visualize first few samples
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        if batch_idx == 0:
            for j in range(min(len(inputs), 5)):
                input_img = denormalize(inputs[j, :3].cpu(), mean, std).numpy().transpose(1, 2, 0)
                target_img = targets[j].cpu().numpy().squeeze()
                output_img = outputs[j].cpu().numpy().squeeze()

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(np.clip(input_img, 0, 1))
                axes[0].set_title('Satellite')

                im1 = axes[1].imshow(target_img, cmap='hot')
                axes[1].set_title('Ground Truth')
                cbar1 = fig.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
                cbar1.set_label('Intensity')

                im2 = axes[2].imshow(output_img, cmap='hot')
                axes[2].set_title('Prediction')
                cbar2 = fig.colorbar(im2, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)
                cbar2.set_label('Intensity')
                                        
                for ax in axes:
                    ax.axis('off')

                plt.savefig(os.path.join(RESULTS_DIR, f"result_epoch_{self.current_epoch}_sample_{j}.png"))
                plt.close()

        return {'iou': iou, 'rmse': rmse}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

def denormalize(img: torch.Tensor, mean: list, std: list) -> torch.Tensor:
    img = img.clone()
    for i in range(img.shape[0]):
        img[i] = img[i] * std[i] + mean[i]
    return img

# Main function
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train or evaluate a segmentation model.')
    parser.add_argument('--model', type=str, choices=['CollisionNet', 'DeepLabV3+', 'UNet++'], default='CollisionNet', help='Choose the model to train.')
    parser.add_argument('--baseline', action='store_true', default=False, help='Train the baseline model.')
    args = parser.parse_args()

    train_dataset = CollisionDataset(root_dir=TRAIN_DATA_DIR, augment=True, baseline=args.baseline)
    test_dataset = CollisionDataset(root_dir=TEST_DATA_DIR, augment=False, baseline=args.baseline)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)

    if args.baseline:
        in_channels = 4
    else:
        in_channels = 5
    # Select model based on argument
    if args.model == 'CollisionNet':
        model = CollisionNet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=in_channels, classes=1)
    elif args.model == 'DeepLabV3+':
        model = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights="imagenet", in_channels=in_channels, classes=1)
    elif args.model == 'UNet++':
        model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights="imagenet", in_channels=in_channels, classes=1)

    # Set RESULTS_DIR according to the model name
    if args.baseline:
        RESULTS_DIR = f"results_{args.model}_baseline_2"
    else:
        RESULTS_DIR = f"results_{args.model}"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Initialize the Lightning module
    pl_model = CollisionNetLightning(model)

    # Create a trainer
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS, check_val_every_n_epoch=10)

    # Train the model
    trainer.fit(pl_model, train_loader, test_loader)