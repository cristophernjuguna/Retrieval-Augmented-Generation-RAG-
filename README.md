# Retrieval-Augmented-Generation-RAG-
🎯 Objective
Complete Retrieval-Augmented Generation (RAG) pipeline for querying Kenyan Finance Bill 2025 using ChromaDB vector database. Extracts text from PDFs, chunks documents, generates semantic embeddings, stores in persistent vector DB, and enables similarity search.

🛠 Tech Stack
text
PDF Processing: pypdf
Embedding Model: BAAI/bge-small-en-v1.5 (384-dim)
Vector Database: ChromaDB (HNSW indexing)
Chunk Size: 800 words (no overlap)
Storage: Persistent SQLite + embeddings
🚀 Step-by-Step Pipeline
1. PDF → Text Extraction
python
from pypdf import PdfReader
def extract_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:      # Loops 40+ pages
        text += page.extract_text() # OCR + layout parsing
    return text  # ~43K words
Output: Raw Finance Bill text (tax amendments, schedules, rates)

2. Text → Chunks (54 chunks)
python
def chunk_text(text, size=800):
    words = text.split()           # 43,219 words
    return [' '.join(words[i:i+size]) for i in range(0, len(words), size)]
Result: 54 non-overlapping chunks covering Income Tax, VAT, Schedules

3. Chunks → Embeddings (54 × 384-dim vectors)
python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
embeddings = model.encode(chunks, normalize_embeddings=True)
Math: L2-normalized vectors → cosine_sim(A,B) = dot_product(A,B)

4. ChromaDB Storage
python
import chromadb
client = chromadb.PersistentClient("./chromadb")
collection = client.get_or_create_collection("finance_collection")
collection.add(documents=chunks, embeddings=embeddings, ids=[f"doc{i}" for i in range(54)])
Storage: ./chromadb/ folder (SQLite + HNSW index, ~5MB total)

5. Semantic Query
python
query = "what does the finance bill say about taxes"
results = collection.query(
    query_embeddings=model.encode([query]), 
    n_results=3
)
Output: Top-3 chunks + cosine distances (0.48, 0.50, 0.52 → lower=better)
