import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def build_image_index():
    print("=" * 60)
    print("Building CRI Reference Image FAISS Vector Index (image_index)")
    print("=" * 60)

    # 1. Resolve paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try finding images_metadata.json in backend/static/images or root
    metadata_path = os.path.join(base_dir, "backend", "static", "images", "images_metadata.json")
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(base_dir, "images_metadata.json")
        
    image_index_dir = os.path.join(base_dir, "image_index")

    if not os.path.exists(metadata_path):
        print(f"ERROR: Metadata file not found at {metadata_path}")
        return False

    print(f"Loading image metadata from: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        image_metadata = json.load(f)

    print(f"Found {len(image_metadata)} image entries.")

    # 2. Convert each metadata entry to Document
    documents = []
    for item in image_metadata:
        filename = item.get("filename", "")
        description = item.get("description", "")
        caption = item.get("caption", "")
        source = item.get("source", "")
        
        # Clean URL format for static file serving
        url = f"/static/images/{filename}"

        doc = Document(
            page_content=description,
            metadata={
                "filename": filename,
                "caption": caption,
                "source": source,
                "url": url
            }
        )
        documents.append(doc)

    print(f"Created {len(documents)} document objects for embedding.")

    # 3. Create embeddings using HuggingFaceEmbeddings all-MiniLM-L6-v2
    print("\nInitializing Sentence-BERT embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    # 4. Build FAISS vector store
    print("\nBuilding FAISS vector index...")
    vector_store = FAISS.from_documents(documents, embeddings)

    # 5. Save local FAISS index
    os.makedirs(image_index_dir, exist_ok=True)
    vector_store.save_local(image_index_dir)

    # Copy to backend/image_index for backend deployment consistency
    backend_index_dir = os.path.join(base_dir, "backend", "image_index")
    os.makedirs(backend_index_dir, exist_ok=True)
    vector_store.save_local(backend_index_dir)

    # Sync metadata files if needed
    root_metadata_path = os.path.join(base_dir, "images_metadata.json")
    if metadata_path != root_metadata_path:
        with open(root_metadata_path, "w", encoding="utf-8") as f:
            json.dump(image_metadata, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("SUCCESS! Image FAISS vector index built successfully.")
    print(f"Saved to: {image_index_dir} and {backend_index_dir}")
    print(f"Total images indexed: {len(documents)}")
    print("=" * 60)
    return True

if __name__ == "__main__":
    build_image_index()
