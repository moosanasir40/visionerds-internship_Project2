import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from app.rag import query_rag
from app.database import get_review_from_db

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY", "")

# Point the OpenAI client to Groq's API
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key
)

def call_groq(messages: list, temperature: float = 0.1) -> str:
    """Helper to call Groq's high-speed Llama models."""
    if not groq_api_key or "your_" in groq_api_key:
        raise ValueError("Missing or invalid GROQ_API_KEY in .env file.")

    # Using Groq's fast Llama 3.3 model
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

def check_claim_tool(claim_text: str) -> dict:
    """Tool: Validates a single specific statement/claim against reference documents."""
    context = query_rag(claim_text, top_k=3)
    
    prompt = f"""
    You are an AI Compliance & QA Specialist. Validate if the claim is supported or false according to official documentation.
    
    Reference Guidelines:
    {context}
    
    Claim to Verify:
    "{claim_text}"
    
    Analyze the claim directly. State clearly whether it is valid or violates policy, and cite the source document.
    """
    
    evaluation = call_groq([{"role": "user", "content": prompt}])
    
    return {
        "claim": claim_text,
        "evaluation": evaluation,
        "reference_sources": context
    }

def load_saved_review_tool(session_id: str) -> str:
    """Tool: Loads previously saved QA review results from SQLite."""
    data = get_review_from_db(session_id)
    if not data:
        return f"No previous QA review found for session '{session_id}'."
    
    issue_lines = "\n".join([f"- [{i['severity']}] {i['type']}: {i['reason']} (Source: {i['source']})" for i in data['issues']])
    return f"Saved Review for Session: {session_id}\nStatus: {data['status']}\nSummary: {data['summary']}\nIssues Found:\n{issue_lines if issue_lines else 'No issues.'}"

def run_document_review(draft_text: str) -> dict:
    """Performs full document QA review and ensures strict JSON response format."""
    context = query_rag(draft_text, top_k=6)
    
    prompt = f"""
    You are an AI Document QA Reviewer. Review the provided draft against official product facts, writing guidelines, and QA rubric.
    
    Rules:
    1. Flag factual errors (e.g., false feature limits or unsupported numerical claims).
    2. Flag compliance/security policy violations.
    3. Do NOT flag text that is factually accurate and compliant.
    4. Set status to 'pass' if no major issues, or 'needs_revision' if issues exist.

    Reference Material:
    {context}

    Draft to Review:
    {draft_text}

    Return ONLY a raw JSON object matching this schema:
    {{
      "status": "pass or needs_revision",
      "summary": "High level review summary",
      "issues": [
        {{
          "type": "Factual Error or Policy Violation or Writing Style",
          "severity": "High or Medium or Low",
          "reason": "Clear explanation of why this was flagged",
          "source": "Name of source reference document"
        }}
      ]
    }}
    """

    content = call_groq([{"role": "user", "content": prompt}], temperature=0.1)
    
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        parsed_json = json.loads(content)
        if "status" not in parsed_json or "issues" not in parsed_json:
            raise ValueError("Missing mandatory keys in review JSON")
        return parsed_json
    except Exception as e:
        return {
            "status": "needs_revision",
            "summary": f"Review completed with formatting fallback: {str(e)}",
            "issues": [{
                "type": "Formatting Check",
                "severity": "Low",
                "reason": content,
                "source": "Model Output"
            }]
        }