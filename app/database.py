import sqlite3
from typing import Dict, Any, List, Optional

DB_NAME = "database.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Store chat history per session
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Store QA reviews
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            document_name TEXT,
            status TEXT,
            summary TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Store specific issues detected in reviews
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS issues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            issue_type TEXT,
            severity TEXT,
            reason TEXT,
            source TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_chat_message(session_id: str, role: str, message: str):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (session_id, role, message) VALUES (?, ?, ?)",
        (session_id, role, message)
    )
    conn.commit()
    conn.close()

def get_chat_history(session_id: str, limit: int = 6) -> List[Dict[str, str]]:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, message FROM chat_history WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    # Reverse to keep chronological order
    return [{"role": r[0], "message": r[1]} for r in reversed(rows)]

def save_review_to_db(session_id: str, doc_name: str, status: str, summary: str, issues: List[Dict[str, Any]]):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO reviews (session_id, document_name, status, summary) VALUES (?, ?, ?, ?)",
        (session_id, doc_name, status, summary)
    )
    for issue in issues:
        cursor.execute(
            "INSERT INTO issues (session_id, issue_type, severity, reason, source) VALUES (?, ?, ?, ?, ?)",
            (
                session_id,
                issue.get("type", "General"),
                issue.get("severity", "Medium"),
                issue.get("reason", ""),
                issue.get("source", "")
            )
        )
    conn.commit()
    conn.close()

def get_review_from_db(session_id: str) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT status, summary FROM reviews WHERE session_id = ? ORDER BY id DESC LIMIT 1",
        (session_id,)
    )
    review = cursor.fetchone()
    if not review:
        conn.close()
        return None

    cursor.execute(
        "SELECT issue_type, severity, reason, source FROM issues WHERE session_id = ?",
        (session_id,)
    )
    issues = cursor.fetchall()
    conn.close()

    return {
        "status": review[0],
        "summary": review[1],
        "issues": [
            {"type": i[0], "severity": i[1], "reason": i[2], "source": i[3]}
            for i in issues
        ]
    }