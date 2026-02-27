import os
import numpy as np
import pandas as pd
from PIL import Image

def generate_dummy_data(base_dir, num_samples=100):
    """
    Generates dummy data simulating the CelebA Face Recognition Triplets structure.

    Args:
        base_dir (str): Base directory where the dummy data will be created.
        num_samples (int): Number of triplet samples to generate.
    """

    # Define directories
    data_dir = os.path.join(base_dir, 'images')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    csv_file = os.path.join(base_dir, 'train.csv')

    # Generate random images
    image_names = []
    for i in range(num_samples * 3): # Generate enough for Anchor, Positive, Negative
        img_name = f"img_{i:04d}.jpg"
        img_path = os.path.join(data_dir, img_name)

        # Create a random image (100x100 RGB)
        img_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(img_path)
        image_names.append(img_name)

    # Create triplets
    triplets = []
    for i in range(num_samples):
        # Simply take 3 consecutive images as a triplet
        anchor = image_names[i*3]
        positive = image_names[i*3+1]
        negative = image_names[i*3+2]
        triplets.append([anchor, positive, negative])

    # Create DataFrame
    df = pd.DataFrame(triplets, columns=['Anchor', 'Positive', 'Negative'])

    # Save CSV
    df.to_csv(csv_file, index=False)
    print(f"Generated {num_samples} dummy triplets in {base_dir}")
    print(f"Images saved to {data_dir}")
    print(f"CSV saved to {csv_file}")

if __name__ == "__main__":
    generate_dummy_data('dummy_celeba_data')
