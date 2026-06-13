# download_models.py
import os
import requests
from tqdm import tqdm

# Model files and their download URLs
# Option A: Use Hugging Face (recommended)
MODEL_URLS = {
    "models/MobileNetV2_best.pth": "https://huggingface.co/spaces/Mansoorrr/wildfire-detection-ai/resolve/main/models/MobileNetV2_best.pth",
    "models/MobileNetV3_best.pth": "https://huggingface.co/spaces/Mansoorrr/wildfire-detection-ai/resolve/main/models/MobileNetV3_best.pth",
    "models/EfficientNet-B0_best.pth": "https://huggingface.co/spaces/Mansoorrr/wildfire-detection-ai/resolve/main/models/EfficientNet-B0_best.pth",
}

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                pbar.update(len(data))

# Create models directory
os.makedirs("models", exist_ok=True)

# Download each model
for filename, url in MODEL_URLS.items():
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        download_file(url, filename)
        print(f"✅ Downloaded {filename}")
    else:
        print(f"✅ {filename} already exists")
