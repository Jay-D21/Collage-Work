import torch
import torch.nn as nn
import timm
import pandas as pd
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import os
import argparse

# Constants
EMB_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DATA_DIR = 'dummy_celeba_data/images'
DEFAULT_CSV_FILE = 'dummy_celeba_data/train.csv'

# Model Definition (Must match training)
class APN_Model(nn.Module):
    def __init__(self, emb_size=EMB_SIZE):
        super(APN_Model, self).__init__()
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True)
        self.efficientnet.classifier = nn.Linear(in_features=self.efficientnet.classifier.in_features, out_features=emb_size)

    def forward(self, images):
        embeddings = self.efficientnet(images)
        return embeddings

def get_encoding(model, img_path):
    img = io.imread(img_path)
    # Normalize and permute
    img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    img = img.unsqueeze(0) # Add batch dimension
    img = img.to(DEVICE)

    model.eval()
    with torch.no_grad():
        enc = model(img)
    return enc.cpu().detach().numpy()

def euclidean_dist(img_enc, anc_enc_arr):
    dist = np.sqrt(np.sum((img_enc - anc_enc_arr)**2, axis=1))
    return dist

def main():
    parser = argparse.ArgumentParser(description="Evaluate CelebA Triplet Network")
    parser.add_argument('--model_path', type=str, default='best_model.pt', help='Path to saved model')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR, help='Directory containing images')
    parser.add_argument('--csv_file', type=str, default=DEFAULT_CSV_FILE, help='Path to CSV file')
    args = parser.parse_args()

    # Load Model
    print(f"Loading model from {args.model_path}")
    model = APN_Model()
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    except FileNotFoundError:
        print("Model file not found. Please train the model first.")
        return
    model.to(DEVICE)

    # Load Database (using the training CSV as the database for this demo)
    print(f"Loading database from {args.csv_file}")
    df = pd.read_csv(args.csv_file)

    # Generate embeddings for all Anchors in the dataset
    print("Generating embeddings for database...")
    database_encodings = []
    database_paths = []

    # Using a subset for speed if needed, but let's do all for the dummy set
    for index, row in df.iterrows():
        img_path = os.path.join(args.data_dir, row['Anchor'])
        try:
            enc = get_encoding(model, img_path)
            database_encodings.append(enc)
            database_paths.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    if not database_encodings:
        print("No embeddings generated.")
        return

    database_encodings = np.vstack(database_encodings)
    print(f"Database shape: {database_encodings.shape}")

    # Pick a random query image from the Positive column to find its Anchor match
    # In a real scenario, this would be a new image.
    # Here, we test if the Positive image retrieves the Anchor image as a close match.

    query_idx = 0 # Random index
    query_img_name = df.iloc[query_idx]['Positive']
    query_img_path = os.path.join(args.data_dir, query_img_name)

    print(f"Querying with image: {query_img_path}")
    query_enc = get_encoding(model, query_img_path)

    # Calculate distances
    distances = euclidean_dist(query_enc, database_encodings)

    # Find closest matches
    closest_indices = np.argsort(distances)[:5]

    print("\nTop 5 matches:")
    for i, idx in enumerate(closest_indices):
        dist = distances[idx]
        path = database_paths[idx]
        print(f"{i+1}: {path} (Distance: {dist:.4f})")

        # Check if it retrieved the correct anchor (simulated identity)
        expected_anchor = df.iloc[query_idx]['Anchor']
        retrieved_anchor = os.path.basename(path)
        if retrieved_anchor == expected_anchor:
             print("   -> MATCH FOUND! (Positive image retrieved its Anchor)")

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
