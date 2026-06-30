from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import fitz         
import pytesseract
from PIL import Image
import io
import os

# Point to Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

KNOWLEDGE_BASE_DIR = "knowledge_base"
FAISS_INDEX_DIR    = "faiss_index"

print("=" * 50)
print("Loading PDFs using OCR (Tesseract)...")
print("=" * 50)

all_documents = []
pdf_files = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.endswith(".pdf")]
print(f"Found {len(pdf_files)} PDF files\n")

for pdf_file in pdf_files:
    pdf_path = os.path.join(KNOWLEDGE_BASE_DIR, pdf_file)
    try:
        doc = fitz.open(pdf_path)
        full_text = ""

        for page_num, page in enumerate(doc):
            # First try normal text extraction
            text = page.get_text()
            if text.strip():
                full_text += text + "\n"
            else:
                # Page is an image — use OCR
                print(f"  OCR on {pdf_file} page {page_num + 1}...")
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes))
                ocr_text = pytesseract.image_to_string(img)
                full_text += ocr_text + "\n"

        if full_text.strip():
            all_documents.append(Document(
                page_content=full_text,
                metadata={"source": pdf_file}
            ))
            print(f"OK: {pdf_file} — {len(full_text)} characters extracted")
        else:
            print(f"SKIP: {pdf_file} — still empty after OCR")

    except Exception as e:
        print(f"ERROR: {pdf_file} — {e}")

print(f"\nTotal documents loaded: {len(all_documents)}")

if not all_documents:
    print("ERROR: No text extracted. Check Tesseract is installed correctly.")
    exit(1)

# Split into chunks
print("\nSplitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(all_documents)
print(f"Created {len(chunks)} chunks")

# Embeddings
print("\nCreating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Build and save FAISS index
print("Building FAISS index...")
vector_store = FAISS.from_documents(chunks, embeddings)
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
vector_store.save_local(FAISS_INDEX_DIR)

print("\n" + "=" * 50)
print("SUCCESS!")
print(f"  PDFs loaded : {len(all_documents)}")
print(f"  Chunks made : {len(chunks)}")
print("=" * 50)
print("\nNext: run  python step2_rag_engine.py")