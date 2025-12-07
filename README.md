# 🧠 AI Photo Search - Backend Server

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-009688?style=for-the-badge&logo=fastapi)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas%2FLocal-47A248?style=for-the-badge&logo=mongodb)
![AI](https://img.shields.io/badge/AI-CLIP%20%2B%20FAISS-orange?style=for-the-badge)

The high-performance, privacy-first API server that powers the AI Photo Search application. It handles semantic image understanding, vector indexing, and user data management without storing actual image files.

---

## 📖 Overview

This backend is a **headless AI engine**. It does not serve a UI; instead, it exposes a RESTful API consumed by mobile or web clients. Its primary responsibility is to bridge the gap between raw pixel data and semantic language queries.

Using **OpenAI's CLIP model**, it converts images and text into high-dimensional vectors. It uses **FAISS** to perform similarity searches on these vectors and **MongoDB** to persist user metadata and embeddings.

### Key Features
* **Stateless Image Processing:** Processes images in memory to generate embeddings and immediately discards the binary data to ensure privacy.
* **Semantic Search Engine:** Enables searching for concepts (e.g., "sunset", "wedding") rather than keywords.
* **Multi-User Architecture:** Supports isolated namespaces for multiple users within a single database.
* **Dynamic Indexing:** Rebuilds vector indexes on-the-fly for real-time search results.

---

## 🛠️ Technical Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Framework** | **FastAPI** | High-performance, async web framework for building APIs. |
| **ML Model** | **CLIP (ViT-B/32)** | Transformer model by OpenAI for multimodal (text/image) embedding. |
| **Vector Search** | **FAISS** | Library for efficient similarity search of dense vectors. |
| **Database** | **MongoDB** | NoSQL database for storing JSON metadata and vector arrays. |
| **Server** | **Uvicorn** | ASGI web server implementation. |

---

## 🚀 Installation & Setup

### 1. Prerequisites
* Python 3.9 or higher
* A MongoDB connection string (Local or MongoDB Atlas)

### 2. Clone and Setup
```bash
# Clone the repository
git clone [https://github.com/yourusername/photo-search-backend.git](https://github.com/yourusername/photo-search-backend.git)
cd photo-search-backend

# Create a virtual environment
python -m venv venv

# Activate environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
