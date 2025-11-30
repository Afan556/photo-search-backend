import argparse
import torch
import numpy as np
import pandas as pd
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import io

# --- Helper Function for L2 Normalization ---
def normalize(v):
    """L2-normalizes a 1D numpy array."""
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

# --- Main Embedding Function ---
def embed_image(image, processor, model, device):
    """
    Computes the L2-normalized embedding for a single PIL image.
    Based on [cite: 92-99]
    """
    # Process the image
    inputs = processor(images=image, return_tensors='pt').to(device)
    
    # Get features
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    
    # Move to CPU, convert to numpy, and get the 1D array
    feats = feats.cpu().numpy()[0]
    
    # L2 normalize the features [cite: 55, 98]
    feats = normalize(feats)
    
    return feats

def main(args):
    # --- 1. Setup ---
    
    # Check for GPU, otherwise use CPU [cite: 28]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directories if they don't exist
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
    Path(args.thumbs).mkdir(parents=True, exist_ok=True)
    
    # --- 2. Load Model ---
    
    # Load the CLIP model and processor from Hugging Face [cite: 54, 91]
    print("Loading CLIP model 'openai/clip-vit-base-patch32'...")
    model = CLIPModel.from_pretrained(args.model_id).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_id)
    model.eval() # Set model to evaluation mode

    # --- 3. Process Images ---
    
    image_dir = Path(args.images)
    # Find all image files recursively
    image_paths = list(image_dir.rglob('*.jpg')) + \
                  list(image_dir.rglob('*.jpeg')) + \
                  list(image_dir.rglob('*.png'))
                  
    print(f"Found {len(image_paths)} images. Processing...")

    all_embeddings = []
    all_metadata = []
    
    # Use tqdm for a progress bar
    for img_path in tqdm(image_paths):
        try:
            # --- 4. Compute Embedding ---
            img = Image.open(img_path).convert('RGB')
            embedding = embed_image(img, processor, model, device)
            all_embeddings.append(embedding)
            
            # --- 5. Create Thumbnail ---
            # Create a thumbnail (e.g., 256px on the longest side)
            img.thumbnail((256, 256))
            thumb_path = Path(args.thumbs) / img_path.name
            img.save(thumb_path, "JPEG")
            
            # --- 6. Store Metadata ---
            # Use relative path for filename
            relative_path = img_path.relative_to(image_dir)
            all_metadata.append({
                "filename": str(relative_path),
                "width": img.width,
                "height": img.height
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # --- 7. Save Outputs ---
    
    # Save embeddings as a numpy file 
    embeddings_array = np.array(all_embeddings, dtype='float32')
    np.save(args.out, embeddings_array)
    print(f"Embeddings saved to {args.out} (Shape: {embeddings_array.shape})")
    
    # Save metadata as a CSV file [cite: 10, 39]
    df = pd.DataFrame(all_metadata)
    df.to_csv(args.meta, index=False)
    print(f"Metadata saved to {args.meta}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed images using CLIP and create thumbnails.")
    
    # Define arguments based on the guide [cite: 57]
    parser.add_argument('--images', type=str, default="images/",
                        help="Path to the directory containing images.")
    parser.add_argument('--out', type=str, default="embeddings.npy",
                        help="Output file path for .npy embeddings.")
    parser.add_argument('--meta', type=str, default="metadata.csv",
                        help="Output file path for .csv metadata.")
    parser.add_argument('--thumbs', type=str, default="thumbnails/",
                        help="Output directory for thumbnails.")
    parser.add_argument('--model-id', type=str, default="openai/clip-vit-base-patch32",
                        help="The CLIP model ID from Hugging Face.")
                        
    args = parser.parse_args()
    main(args)