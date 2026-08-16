# AI Document QA Reviewer (Capstone Project 02)

An intelligent quality assurance reviewer built with **FastAPI**, **ChromaDB**, **Sentence Transformers**, and **Groq (Llama 3.3)**. The system inspects draft product documentation against official product manuals, brand policies, and compliance rubrics to automatically detect factual errors, unsupported claims, and writing guideline violations.

---

## 📌 Features

- **Document QA Review Pipeline:** Automatically evaluates draft content and returns a strictly validated JSON structure with review status (`pass` / `needs_revision`), summaries, and itemized issues with severity levels.
- **Retrieval-Augmented Generation (RAG):** Embeds and indexes official product guides, admin documentation, and compliance policies using `all-MiniLM-L6-v2` and ChromaDB.
- **Conversational Memory & Follow-up Rewriting:** Retains chat history in SQLite and rewrites ambiguous follow-up questions into standalone search queries before querying the vector store.
- **Tool / Action Invocations:** Features dedicated tools to verify standalone claims (`check_claim_tool`) and retrieve stored review states from SQLite (`load_saved_review_tool`).
- **Multi-Path Request Routing:** Routes incoming chat requests across three separate paths: Action Tools, Document Search (RAG), and Standard Assistant Replies.
- **Continuous Integration (CI):** Automated unit test suite run on every commit via GitHub Actions.

---

## 📁 Repository Structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI workflow
├── app/
│   ├── __init__.py
│   ├── database.py                # SQLite persistence for reviews and chat logs
│   ├── main.py                    # FastAPI application, routing, and endpoints
│   ├── rag.py                     # ChromaDB vector store and query rewriting logic
│   └── tools.py                   # Groq LLM integration, tools, and JSON validation
├── supporting_assets/
│   └── 02_ai_document_qa_reviewer/ # Provided reference PDFs
├── tests/
│   └── test_api.py                # Automated Pytest suite
├── .env.example                   # Example environment variables template
├── .gitignore                     # Git ignore file
├── pytest.ini                     # Pytest configuration
├── requirements.txt               # Project dependencies
└── README.md                      # Documentation
⚙️ Installation and Setup1. PrerequisitesPython 3.10+Groq API Key2. Clone the RepositoryBashgit clone [https://github.com/moosanasir40/visionerds-internship_Project2.git](https://github.com/moosanasir40/visionerds-internship_Project2.git)
cd visionerds-internship_Project2
3. Set Up Virtual EnvironmentBash# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
4. Install DependenciesBashpip install -r requirements.txt
5. Environment VariablesCreate a .env file in the root folder:Code snippetGROQ_API_KEY=gsk_your_groq_api_key_here
🚀 Running the ApplicationStart the FastAPI local development server:Bashuvicorn app.main:app --reload
Interactive API documentation (Swagger UI) is available at:👉 http://127.0.0.1:8000/docs📡 API EndpointsMethodEndpointDescriptionPOST/reviewPerforms a full QA check on a draft text and saves the result to SQLite.POST/check-claimEvaluates a single specific claim or numeric metric against reference docs.POST/chatMulti-path conversational endpoint supporting tools, RAG lookups, and context memory.GET/history/{session_id}Retrieves the chronological chat history for a session.🧪 Running Tests & CIRun the automated test suite locally:Bashpython -m pytest
GitHub Actions automatically executes this test suite on every push or pull_request to ensure system stability.🎬 Demo CasesCatch False Unlimited Retention Claim:Call POST /check-claim with "NovaFlow Starter plan provides unlimited reporting retention history."Result: Correctly identifies that the Starter plan is limited to 30 days of reporting history.Catch Unsupported 60% Cost Reduction Claim:Call POST /check-claim with "FieldTrack Ops guarantees a 60 percent reduction in maintenance costs."Result: Flags the statement as an unsupported numerical claim per the Brand and Writing Guide.Verify Valid Content:Call POST /review with accurate facts from the LedgerLane billing guide.Result: Returns "status": "pass" without false positive flags.Conversational Follow-up with Source Attribution:Call POST /chat with "Why was the retention claim flagged and what is the source?"Result: Rewrites query using conversation memory and returns source citations from official documents.Load Review State (Tool Invocation):Call POST /chat with "load review" using an existing session_id.Result: Invokes the SQLite action tool to return the previously saved QA report.
