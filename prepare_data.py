import os
import urllib.request
import numpy as np
from tqdm import tqdm

def download_file(url, dest):
    print(f"Downloading {url} to {dest}")
    if os.path.exists(dest) and os.path.getsize(dest) > 1000000:
        print(f"{dest} already exists and is large, skipping download.")
        return
    urllib.request.urlretrieve(url, dest)
    print(f"Downloaded {dest}")

def extract_and_split():
    base_dir = "/Users/sumankar/Desktop/WS_EMCAD/data/covid19/train"
    images_dir = os.path.join(base_dir, "images")
    masks_dir = os.path.join(base_dir, "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    data_url = "https://huggingface.co/datasets/aryadomain/Medical-2D-Dataset/resolve/main/covid19/data_train.npy"
    mask_url = "https://huggingface.co/datasets/aryadomain/Medical-2D-Dataset/resolve/main/covid19/mask_train.npy"

    data_dest = "/tmp/data_train.npy"
    mask_dest = "/tmp/mask_train.npy"

    download_file(data_url, data_dest)
    download_file(mask_url, mask_dest)

    print("Loading data_train.npy...")
    data = np.load(data_dest, allow_pickle=True)
    print("Loading mask_train.npy...")
    mask = np.load(mask_dest, allow_pickle=True)

    print(f"Data shape: {data.shape}, Mask shape: {mask.shape}")
    
    # Save each slice as a separate npy file
    for i in tqdm(range(len(data))):
        img_slice = data[i]
        mask_slice = mask[i]
        
        np.save(os.path.join(images_dir, f"slice_{i:04d}.npy"), img_slice)
        np.save(os.path.join(masks_dir, f"slice_{i:04d}.npy"), mask_slice)
        
    print("Extraction and split complete.")

if __name__ == "__main__":
    extract_and_split()
