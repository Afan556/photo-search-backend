import torch
import numpy as np
import faiss
from transformers import CLIPProcessor, CLIPModel
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import argparse
from pathlib import Path
from PIL import Image
import io
import base64
from datetime import datetime
import pymongo
from bson import ObjectId
import os

# --- Helper Functions ---

def normalize(v):
    """L2-normalizes a 1D numpy array."""
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

def embed_image(image, processor, model, device):
    """Computes the L2-normalized embedding for a PIL image."""
    inputs = processor(images=image, return_tensors='pt').to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = feats.cpu().numpy()[0]
    return normalize(feats)

def embed_text(text: str, processor, model, device):
    """Computes the L2-normalized embedding for a text query."""
    inputs = processor(text=[text], return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        text_feats = model.get_text_features(**inputs)
    q = text_feats.cpu().numpy()[0]
    return normalize(q)

# --- Pydantic Models ---

class ImageUploadResponse(BaseModel):
    image_id: str
    filename: str
    message: str

class SearchResult(BaseModel):
    image_id: str
    local_path: str
    filename: str
    score: float
    timestamp: str

class UserInit(BaseModel):
    user_id: str
    device_id: str

# --- FastAPI App ---

app = FastAPI(title="Photo Search API")

# Enable CORS for Android app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
models = {}

# --- MongoDB Setup ---

def get_mongo_client():
    """Connect to MongoDB Atlas (Free Tier)"""
    # Replace with your MongoDB connection string
    # Sign up at: https://www.mongodb.com/cloud/atlas/register
    MONGO_URI = os.getenv("MONGO_URI",  "mongodb+srv://username:password@cluster.mongodb.net/")
    client = pymongo.MongoClient(MONGO_URI)
    return client

@app.on_event("startup")
def startup_event():
    """Load models and connect to database on startup"""
    print("🚀 Server starting up...")
    
    # Device setup
    models["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📱 Using device: {models['device']}")

    # Load CLIP Model
    print("🤖 Loading CLIP model...")
    model_id = "openai/clip-vit-base-patch32"
    models["clip_model"] = CLIPModel.from_pretrained(model_id).to(models["device"])
    models["clip_processor"] = CLIPProcessor.from_pretrained(model_id)
    models["clip_model"].eval()
    
    # MongoDB connection
    print("💾 Connecting to MongoDB...")
    models["mongo_client"] = get_mongo_client()
    models["db"] = models["mongo_client"]["photo_search_db"]
    models["images_collection"] = models["db"]["images"]
    
    # Create index on user_id for faster queries
    models["images_collection"].create_index("user_id")
    
    print("✅ Startup complete. Server is ready!")

@app.on_event("shutdown")
def shutdown_event():
    """Close MongoDB connection on shutdown"""
    if "mongo_client" in models:
        models["mongo_client"].close()
        print("🔌 MongoDB connection closed.")

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {
        "message": "Photo Search API",
        "version": "2.0",
        "endpoints": {
            "POST /init": "Initialize user",
            "POST /upload": "Upload image for embedding",
            "POST /upload_batch": "Upload multiple images",
            "GET /search": "Search images by text",
            "DELETE /image/{image_id}": "Delete an image",
            "GET /stats": "Get user statistics"
        }
    }

@app.post("/init")
def initialize_user(user_data: UserInit):
    """Initialize a new user or return existing user stats"""
    user_id = user_data.user_id
    
    # Count existing images for this user
    image_count = models["images_collection"].count_documents({"user_id": user_id})
    
    return {
        "user_id": user_id,
        "image_count": image_count,
        "message": "User initialized successfully"
    }

@app.post("/upload", response_model=ImageUploadResponse)
async def upload_image(
    file: UploadFile = File(...),
    user_id: str = Query(..., description="User ID from Android device"),
    local_path: str = Query(..., description="Local file path on Android device")
):
    """
    Upload a single image for embedding generation.
    The actual image stays on the device, we only store the embedding.
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Generate embedding
        embedding = embed_image(
            image, 
            models["clip_processor"], 
            models["clip_model"], 
            models["device"]
        )
        
        # Prepare document for MongoDB
        image_doc = {
            "user_id": user_id,
            "filename": file.filename,
            "local_path": local_path,  # Android device path
            "embedding": embedding.tolist(),  # Store as list for MongoDB
            "width": image.width,
            "height": image.height,
            "uploaded_at": datetime.utcnow(),
        }
        
        # Insert into MongoDB
        result = models["images_collection"].insert_one(image_doc)
        
        return ImageUploadResponse(
            image_id=str(result.inserted_id),
            filename=file.filename,
            message="Image processed successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/upload_batch")
async def upload_batch(
    files: List[UploadFile] = File(...),
    user_id: str = Query(...),
    local_paths: str = Query(..., description="Comma-separated local paths")
):
    """Upload multiple images at once for faster processing"""
    local_paths_list = local_paths.split(",")
    
    if len(files) != len(local_paths_list):
        raise HTTPException(status_code=400, detail="Number of files and paths must match")
    
    results = []
    
    for file, local_path in zip(files, local_paths_list):
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            
            embedding = embed_image(
                image, 
                models["clip_processor"], 
                models["clip_model"], 
                models["device"]
            )
            
            image_doc = {
                "user_id": user_id,
                "filename": file.filename,
                "local_path": local_path.strip(),
                "embedding": embedding.tolist(),
                "width": image.width,
                "height": image.height,
                "uploaded_at": datetime.utcnow(),
            }
            
            result = models["images_collection"].insert_one(image_doc)
            results.append({
                "image_id": str(result.inserted_id),
                "filename": file.filename,
                "status": "success"
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "failed",
                "error": str(e)
            })
    
    return {"results": results, "total": len(files), "successful": sum(1 for r in results if r.get("status") == "success")}

@app.get("/search", response_model=List[SearchResult])
def search(
    q: str = Query(..., description="Text query to search for"),
    user_id: str = Query(..., description="User ID"),
    k: int = Query(10, description="Number of results", gt=0, le=50)
):
    """
    Search for images using text query.
    Returns local paths so the app can load images from device storage.
    """
    try:
        # Get query embedding
        query_vector = embed_text(
            q, 
            models["clip_processor"], 
            models["clip_model"], 
            models["device"]
        )
        
        # Fetch all user's images from MongoDB
        user_images = list(models["images_collection"].find({"user_id": user_id}))
        
        if not user_images:
            return []
        
        # Build FAISS index on-the-fly
        embeddings = np.array([img["embedding"] for img in user_images], dtype='float32')
        dimension = embeddings.shape[1]
        
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        # Search
        query_vector_2d = np.array([query_vector], dtype='float32')
        scores, indices = index.search(query_vector_2d, min(k, len(user_images)))
        
        # Build results
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            score = float(scores[0][i])
            
            img_doc = user_images[idx]
            
            results.append(SearchResult(
                image_id=str(img_doc["_id"]),
                local_path=img_doc["local_path"],
                filename=img_doc["filename"],
                score=score,
                timestamp=img_doc["uploaded_at"].isoformat()
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.delete("/image/{image_id}")
def delete_image(image_id: str, user_id: str = Query(...)):
    """Delete an image embedding from the database"""
    try:
        result = models["images_collection"].delete_one({
            "_id": ObjectId(image_id),
            "user_id": user_id
        })
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Image not found")
        
        return {"message": "Image deleted successfully", "image_id": image_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete error: {str(e)}")

@app.get("/stats")
def get_stats(user_id: str = Query(...)):
    """Get statistics for a user"""
    try:
        image_count = models["images_collection"].count_documents({"user_id": user_id})
        
        # Get latest upload time
        latest = models["images_collection"].find_one(
            {"user_id": user_id},
            sort=[("uploaded_at", -1)]
        )
        
        return {
            "user_id": user_id,
            "total_images": image_count,
            "last_upload": latest["uploaded_at"].isoformat() if latest else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

# --- Run Server ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Photo Search API server")
    parser.add_argument('--host', type=str, default="0.0.0.0")
    parser.add_argument('--port', type=int, default=8000)
    
    args = parser.parse_args()
    
    print(f"🌐 Starting server at http://{args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port)
