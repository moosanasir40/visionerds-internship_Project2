from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from app.database import (
    init_db,
    save_chat_message,
    get_chat_history,
    save_review_to_db,
    get_review_from_db
)
from app.rag import load_documents, query_rag, rewrite_query_if_vague
from app.tools import run_document_review, check_claim_tool, load_saved_review_tool

app = FastAPI(
    title="AI Document QA Reviewer API",
    description="Automated QA review agent using RAG and rule enforcement",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    init_db()
    load_documents()

# Pydantic Schemas
class ChatRequest(BaseModel):
    session_id: str = Field(..., example="qa-session-001")
    message: str = Field(..., example="Why was issue 1 flagged?")

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    path_taken: str

class ReviewRequest(BaseModel):
    session_id: str = Field(..., example="qa-session-001")
    document_name: Optional[str] = "draft_to_review.pdf"
    document_text: str

class ClaimCheckRequest(BaseModel):
    claim: str = Field(..., example="NovaFlow Starter plan includes unlimited reporting history.")

# Endpoints

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    Handles conversational interactions across 3 execution paths:
    Path 1: Tool / Action Execution (load review)
    Path 2: RAG Search with Query Rewriting (why, source, rule queries)
    Path 3: Normal conversational reply
    """
    try:
        msg_clean = req.message.strip()
        msg_lower = msg_clean.lower()
        
        # Fetch conversation memory
        history = get_chat_history(req.session_id)
        history_text = "\n".join([f"{h['role']}: {h['message']}" for h in history])

        # Path 1: Tool / Action invocation
        if "load review" in msg_lower or "saved review" in msg_lower:
            reply = load_saved_review_tool(req.session_id)
            path = "tool_action:load_saved_review"

        # Path 2: Document Search / Follow-up Q&A
        elif any(keyword in msg_lower for keyword in ["why", "source", "rule", "policy", "explain", "flag", "evidence"]):
            # Rewrite query if it depends on conversation context
            standalone_query = rewrite_query_if_vague(msg_clean, history_text)
            context = query_rag(standalone_query, top_k=3)
            reply = f"Here is the relevant source rule and evidence:\n\n{context}"
            path = "rag_document_search"

        # Path 3: Normal Assistant Reply
        else:
            reply = (
                f"Hello! I am ready to review documents for session '{req.session_id}'. "
                "You can submit text to POST /review, test a specific claim with POST /check-claim, "
                "or ask me questions about the review results."
            )
            path = "normal_reply"

        # Save to SQLite conversation memory
        save_chat_message(req.session_id, "user", req.message)
        save_chat_message(req.session_id, "assistant", reply)

        return ChatResponse(session_id=req.session_id, reply=reply, path_taken=path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.post("/review")
def review_endpoint(req: ReviewRequest):
    """Performs QA checks on draft text, stores issues in SQLite, and returns JSON."""
    try:
        result = run_document_review(req.document_text)
        
        save_review_to_db(
            session_id=req.session_id,
            doc_name=req.document_name or "draft_doc",
            status=result.get("status", "needs_revision"),
            summary=result.get("summary", ""),
            issues=result.get("issues", [])
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Review failed: {str(e)}")

@app.post("/check-claim")
def check_claim_endpoint(req: ClaimCheckRequest):
    """Tool-powered endpoint to quickly verify a single statement."""
    try:
        return check_claim_tool(req.claim)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claim check failed: {str(e)}")

@app.get("/history/{session_id}")
def get_history_endpoint(session_id: str):
    """Retrieve chat history for a session."""
    return {"session_id": session_id, "history": get_chat_history(session_id)}