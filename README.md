# 📸 AI Photo Search (Privacy-First)

![Android](https://img.shields.io/badge/Client-Android%20(Kotlin)-green?style=for-the-badge&logo=android)
![Backend](https://img.shields.io/badge/Backend-FastAPI%20(Python)-blue?style=for-the-badge&logo=python)
![Database](https://img.shields.io/badge/DB-MongoDB%20%2B%20FAISS-forestgreen?style=for-the-badge&logo=mongodb)
![AI](https://img.shields.io/badge/Model-OpenAI%20CLIP-orange?style=for-the-badge&logo=openai)

A self-hosted, semantic search engine for your personal photo gallery. Think of it as **"Google Photos, but completely private."**

Instead of uploading your photos to a big tech cloud, this app processes them locally or on your own private server. It uses **OpenAI's CLIP model** to understand the content of your images and **FAISS** for ultra-fast vector search, allowing you to find photos by describing them (e.g., *"cat on the grass"* or *"wedding cake"*).

---

## 📱 Features

* **🔍 Natural Language Search:** Search for photos using descriptions, objects, or actions. No tags required.
* **🔒 Privacy-First Architecture:** Your actual photo files never leave your device/local network. Only mathematical embeddings are stored.
* **☁️ Smart Sync:** Automatically scans your Android gallery and syncs vector embeddings to the database.
* **👤 Multi-User Support:** Supports multiple users with isolated galleries on the same backend.
* **⚡ Real-Time Progress:** Visual progress tracking during gallery synchronization.
* **📤 Instant Sharing:** Share found photos directly to WhatsApp, Instagram, or Email.

---

## 🏗️ System Architecture

This project follows a decoupled **Client-Server Architecture**:

1.  **The Client (Android/Kotlin):**
    * Scans local storage for images.
    * Sends images to the server *only* for embedding generation.
    * Displays search results using local file paths (Zero latency, full privacy).
2.  **The Server (Python/FastAPI):**
    * **API Layer:** FastAPI handles upload and search requests.
    * **AI Engine:** Uses `transformers` (CLIP) to convert images/text into 512-dimensional vectors.
    * **Vector DB:** Uses **FAISS** (Facebook AI Similarity Search) to index vectors for millisecond-level retrieval.
    * **Metadata DB:** Uses **MongoDB** to store user profiles, file paths, and metadata.

---

## 🛠️ Tech Stack

### Android Client
* **Language:** Kotlin
* **Networking:** Retrofit + OkHttp (with timeout handling for large uploads)
* **Image Loading:** Coil (Coroutines Image Loader)
* **Concurrency:** Kotlin Coroutines & Lifecycle Scopes
* **UI:** Material Design, CardViews, Custom Drawables

### Python Backend
* **Framework:** FastAPI + Uvicorn
* **ML Model:** OpenAI CLIP (`clip-vit-base-patch32`)
* **Vector Search:** FAISS (IndexFlatIP for exact L2 search)
* **Database:** MongoDB (via `pymongo`)
* **Image Processing:** Pillow (PIL)

---

## 🚀 Installation & Setup

### Prerequisites
* Python 3.9+
* Android Studio Iguana (or newer)
* MongoDB Connection String (Atlas or Local)

### 1. Backend Setup (The Server)

```bash
# Clone the repository
git clone [https://github.com/yourusername/photo-search-fyp.git](https://github.com/yourusername/photo-search-fyp.git)
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision transformers fastapi uvicorn pymongo faiss-cpu python-multipart pillow

# Run the server
# Make sure to allow firewall access if prompted!
python main_server.py
