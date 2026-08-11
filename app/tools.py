from rag import search_docs
from db import save_review_db

def check_rule(query: str) -> str:
    """Tool 1: Searches product facts, writing rules, and QA rubric."""
    results = search_docs(query, n_results=2)
    if not results:
        return "No relevant rule or fact found in reference docs."
    
    formatted = []
    for r in results:
        formatted.append(f"[Source: {r['source']}]\n{r['content']}")
    return "\n\n".join(formatted)

def save_review(session_id: str, status: str, issues: list, summary: str) -> str:
    """Tool 2: Saves review results into SQLite database."""
    save_review_db(session_id, status, issues, summary)
    return f"Successfully saved review for session '{session_id}' to database."