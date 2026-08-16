import os
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="qa_reference_docs")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def load_documents(folder_path: str = "supporting_assets/02_ai_document_qa_reviewer"):
    """Reads all PDF reference guides and indexes them into ChromaDB."""
    if collection.count() > 0:
        return

    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    doc_id = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(root, file)
                try:
                    reader = PdfReader(file_path)
                    text = ""
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
                    
                    # Split into meaningful chunks
                    chunks = [c.strip() for c in text.split("\n\n") if len(c.strip()) > 30]
                    for chunk in chunks:
                        embedding = EMBED_MODEL.encode(chunk).tolist()
                        collection.add(
                            documents=[chunk],
                            embeddings=[embedding],
                            metadatas=[{"source": file}],
                            ids=[f"doc_{doc_id}"]
                        )
                        doc_id += 1
                except Exception as e:
                    print(f"Error reading {file}: {e}")

def rewrite_query_if_vague(user_query: str, chat_history_text: str) -> str:
    """Rewrites pronouns or vague follow-up queries using earlier chat context."""
    if not chat_history_text.strip():
        return user_query

    prompt = f"""
    Given the following conversation history and a user follow-up question, rewrite the follow-up into a standalone, specific search query.
    If it is already specific, return the original query unchanged.

    Conversation History:
    {chat_history_text}

    Follow-up Question: {user_query}

    Standalone Query:"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else user_query
    except Exception:
        return user_query

def query_rag(query_text: str, top_k: int = 4) -> str:
    """Queries ChromaDB vector collection and returns formatted context chunks."""
    if collection.count() == 0:
        load_documents()

    query_emb = EMBED_MODEL.encode(query_text).tolist()
    results = collection.query(query_embeddings=[query_emb], n_results=min(top_k, max(1, collection.count())))
    
    if not results or not results["documents"] or not results["documents"][0]:
        return "No relevant reference documentation found."

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    
    context_str = ""
    for doc, meta in zip(docs, metas):
        context_str += f"[Source Document: {meta['source']}]\n{doc}\n\n"
    return context_str