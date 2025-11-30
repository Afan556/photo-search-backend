import numpy as np
import faiss
import argparse
from pathlib import Path

def main(args):
    # --- 1. Load Embeddings ---
    print(f"Loading embeddings from {args.embeddings}...")
    
    # Ensure the file exists before trying to load
    embeddings_path = Path(args.embeddings)
    if not embeddings_path.exists():
        print(f"Error: Embeddings file not found at {embeddings_path}")
        print("Please run embed_images.py first.")
        return

    embeddings = np.load(embeddings_path).astype('float32')
    
    # Get the dimensions of the embeddings
    # (N vectors, D dimensions)
    N, D = embeddings.shape 
    print(f"Loaded {N} embeddings, each with {D} dimensions.")

    # --- 2. Create FAISS Index ---
    
    # We use IndexFlatIP (Inner Product) because our vectors are
    # L2-normalized. Cosine Similarity is equivalent to Inner Product
    # [cite_start]on normalized vectors. [cite: 55, 63, 67]
    
    print("Creating FAISS index (IndexFlatIP)...")
    index = faiss.IndexFlatIP(D)
    
    # --- 3. Add Vectors to Index ---
    print("Adding vectors to the index...")
    index.add(embeddings)
    
    # --- 4. Save Index to Disk ---
    print(f"Saving index to {args.out}...")
    faiss.write_index(index, args.out)
    
    print("\nDone.")
    print(f"Your searchable index is now ready at: {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a FAISS index from .npy embeddings.")
    
    parser.add_argument('--embeddings', type=str, default="embeddings.npy",
                        help="Path to the .npy file containing embeddings.")
    parser.add_argument('--out', type=str, default="index.faiss",
                        help="Output file path for the FAISS index.")
                        
    args = parser.parse_args()
    main(args)