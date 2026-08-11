import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from app.rag import query_rag

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def run_document_review(draft_text: str):
    context = query_rag(draft_text, top_k=5)
    
    prompt = f"""
    You are a strict Document QA Reviewer. Check this draft text against reference rules.
    Reference Facts and Rules:
    {context}

    Draft to Review:
    {draft_text}

    Return ONLY a clean JSON object with this exact structure:
    {{
      "status": "pass or needs_revision",
      "summary": "Short explanation",
      "issues": [
        {{
          "type": "Factual Error / Policy Violation",
          "severity": "High / Medium / Low",
          "reason": "Why it was flagged",
          "source": "Name of source file"
        }}
      ]
    }}
    """

    response = client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content
    
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    parsed_json = json.loads(content)
    return parsed_json