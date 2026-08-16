import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from app.rag import query_rag
from app.database import get_review_from_db

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def check_claim_tool(claim_text: str) -> dict:
    """Tool: Validates a single specific statement/claim against reference documents."""
    context = query_rag(claim_text, top_k=3)
    
    prompt = f"""
    You are an AI Compliance & QA Specialist. Validate if the claim is supported or false.
    
    Reference Guidelines:
    {context}
    
    Claim to Verify:
    "{claim_text}"
    
    Analyze the claim against the facts. Respond clearly whether it is valid or violates policy, citing the source document.
    """
    
    response = client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=[{"role": "user", "content": prompt}]
    )
    return {
        "claim": claim_text,
        "evaluation": response.choices[0].message.content.strip(),
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
    1. Flag factual errors (e.g. false feature limits or unsupported numerical claims).
    2. Flag compliance/security policy violations.
    3. Do NOT flag text that is factually accurate and compliant.
    4. Set status to 'pass' if no major issues, or 'needs_revision' if issues exist.

    Reference Material:
    {context}

    Draft to Review:
    {draft_text}

    Return ONLY a raw JSON object (no markdown, no extra commentary) matching this schema:
    {{
      "status": "pass" or "needs_revision",
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

    response = client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    content = response.choices[0].message.content.strip()
    
    # Strip markdown wrappers if returned
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    # Code validation on JSON output
    try:
        parsed_json = json.loads(content)
        # Ensure mandatory keys exist
        if "status" not in parsed_json or "issues" not in parsed_json:
            raise ValueError("Missing mandatory keys in review JSON")
        return parsed_json
    except Exception as e:
        return {
            "status": "needs_revision",
            "summary": f"Automated review completed with formatting fallback: {str(e)}",
            "issues": [{
                "type": "Parsing Check",
                "severity": "Low",
                "reason": content,
                "source": "Model Output"
            }]
        }