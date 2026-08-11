import os
import glob
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

# Initialize Vector DB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="qa_reference_docs",
    embedding_function=sentence_transformer_ef
)

def load_reference_documents(docs_folder: str = "supporting_assets/02_ai_document_qa_reviewer"):
    """Loads all PDFs in product_docs, policies, and approved_content."""
    if collection.count() > 0:
        return  # Already loaded

    pdf_files = glob.glob(f"{docs_folder}/**/*.pdf", recursive=True)
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        reader = PdfReader(pdf_path)
        
        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                # Store text chunks with metadata
                collection.add(
                    documents=[text],
                    metadatas=[{"source": filename, "page": idx + 1}],
                    ids=[f"{filename}_p{idx+1}"]
                )

def search_docs(query: str, n_results: int = 3):
    """Retrieves context matching the query."""
    results = collection.query(query_texts=[query], n_results=n_results)
    
    retrieved_chunks = []
    if results and results['documents']:
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            retrieved_chunks.append({
                "content": doc,
                "source": meta['source']
            })
    return retrieved_chunks