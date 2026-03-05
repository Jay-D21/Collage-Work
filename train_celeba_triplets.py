import pandas as pd
import numpy as np
import torch
import timm
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from skimage import io
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import argparse

# Define constants
# Default to dummy data, but allow overriding via arguments
DEFAULT_DATA_DIR = 'dummy_celeba_data/images'
DEFAULT_CSV_FILE = 'dummy_celeba_data/train.csv'
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 1
EMB_SIZE = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class APN_Dataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Paths to images
        A_path = os.path.join(self.data_dir, row.Anchor)
        P_path = os.path.join(self.data_dir, row.Positive)
        N_path = os.path.join(self.data_dir, row.Negative)

        # Read images
        A_img = io.imread(A_path)
        P_img = io.imread(P_path)
        N_img = io.imread(N_path)

        # Handle grayscale images if any (convert to RGB if needed, though efficientnet expects 3 channels)
        # Assuming images are RGB for now based on dummy data.
        # If they were grayscale, we'd need: if len(img.shape) == 2: img = np.stack((img,)*3, axis=-1)

        # Normalize and permute: (H, W, C) -> (C, H, W)
        A_img = torch.from_numpy(A_img).permute(2,0,1).float() / 255.0
        P_img = torch.from_numpy(P_img).permute(2,0,1).float() / 255.0
        N_img = torch.from_numpy(N_img).permute(2,0,1).float() / 255.0

        return A_img, P_img, N_img

class APN_Model(nn.Module):
    def __init__(self, emb_size=EMB_SIZE):
        super(APN_Model, self).__init__()
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
        self.efficientnet.classifier = nn.Linear(in_features=self.efficientnet.classifier.in_features, out_features=emb_size)

    def forward(self, images):
        embeddings = self.efficientnet(images)
        return embeddings

def train_fn(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for A, P, N in tqdm(dataloader, desc="Training"):
        A, P, N = A.to(DEVICE), P.to(DEVICE), N.to(DEVICE)

        optimizer.zero_grad()

        A_embs = model(A)
        P_embs = model(P)
        N_embs = model(N)

        loss = criterion(A_embs, P_embs, N_embs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def eval_fn(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for A, P, N in tqdm(dataloader, desc="Validating"):
            A, P, N = A.to(DEVICE), P.to(DEVICE), N.to(DEVICE)
            A_embs = model(A)
            P_embs = model(P)
            N_embs = model(N)

            loss = criterion(A_embs, P_embs, N_embs)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser(description="Train CelebA Triplet Network")
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR, help='Directory containing images')
    parser.add_argument('--csv_file', type=str, default=DEFAULT_CSV_FILE, help='Path to CSV file with triplets')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size')
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print(f"Loading data from {args.csv_file} and {args.data_dir}")

    # Load Data
    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {args.csv_file}. Please run generate_dummy_data.py first or provide correct path.")
        return

    # Split Data
    train_df, valid_df = train_test_split(df, test_size=0.20, random_state=42)

    # Create Datasets and Loaders
    trainset = APN_Dataset(train_df, args.data_dir)
    validset = APN_Dataset(valid_df, args.data_dir)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    validloader = DataLoader(validset, batch_size=args.batch_size)

    print(f"Size of trainset : {len(trainset)}")
    print(f"Size of validset : {len(validset)}")

    # Model, Criterion, Optimizer
    model = APN_Model()
    model.to(DEVICE)

    criterion = nn.TripletMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training Loop
    best_valid_loss = np.inf
    for i in range(args.epochs):
        print(f"Epoch {i+1}/{args.epochs}")
        train_loss = train_fn(model, trainloader, optimizer, criterion)
        valid_loss = eval_fn(model, validloader, criterion)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), 'best_model.pt')
            best_valid_loss = valid_loss
            print("Saved Best Weights")

if __name__ == "__main__":
    main()
