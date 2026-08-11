import os
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="qa_reference_docs")

def load_documents(folder_path="supporting_assets/02_ai_document_qa_reviewer"):
    if collection.count() > 0:
        return

    doc_id = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                
                chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 20]
                for chunk in chunks:
                    embedding = EMBED_MODEL.encode(chunk).tolist()
                    collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        metadatas=[{"source": file}],
                        ids=[f"doc_{doc_id}"]
                    )
                    doc_id += 1

def query_rag(query_text: str, top_k: int = 3):
    query_emb = EMBED_MODEL.encode(query_text).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    
    context_str = ""
    for doc, meta in zip(docs, metas):
        context_str += f"[Source: {meta['source']}]\n{doc}\n\n"
    return context_str