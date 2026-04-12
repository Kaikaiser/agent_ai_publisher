from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import typer

from app.auth.service import AuthService
from app.core.config import get_settings
from app.db.init_db import bootstrap_admin, init_db
from app.db.models import ConversationRecord
from app.db.session import SessionLocal
from app.services.chat import ChatService
from app.services.faithfulness import FaithfulnessEvaluator
from app.services.knowledge import KnowledgeService
from app.services.providers import OpenAICompatibleProvider

cli = typer.Typer(help="Publisher AI Assistant CLI")


@cli.command("create-admin")
def create_admin(username: str, password: str):
    init_db()
    db = SessionLocal()
    try:
        user = AuthService(db).create_admin(username, password)
        typer.echo(f"Created admin: {user.username}")
    finally:
        db.close()


@cli.command("bootstrap-admin")
def bootstrap_default_admin():
    init_db()
    bootstrap_admin()
    typer.echo("Default admin ensured")


@cli.command("import-knowledge")
def import_knowledge(username: str, file: str, book_title: str, doc_type: str, allowed_role: str = "user"):
    init_db()
    db = SessionLocal()
    provider = OpenAICompatibleProvider()
    try:
        service = KnowledgeService(db, provider.get_embeddings())
        job = service.import_file(file, Path(file).name, username, book_title, doc_type, allowed_role)
        typer.echo(f"Import completed: job={job.id} message={job.message}")
    finally:
        db.close()


@cli.command("rebuild-knowledge-index")
def rebuild_knowledge_index():
    init_db()
    db = SessionLocal()
    provider = OpenAICompatibleProvider()
    try:
        result = KnowledgeService(db, provider.get_embeddings()).rebuild_index()
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        db.close()


@cli.command("ask")
def ask(username: str, question: str, role: str = "admin", book_title: str = "", doc_type: str = ""):
    init_db()
    db = SessionLocal()
    provider = OpenAICompatibleProvider()
    try:
        service = ChatService(db, provider.get_llm(), provider.get_embeddings())
        result = service.ask(username, role, question, book_title or None, doc_type or None)
        typer.echo(result["answer"])
    finally:
        db.close()


@cli.command("evaluate-faithfulness")
def evaluate_faithfulness(dataset_path: str = "", report_path: str = ""):
    init_db()
    provider = OpenAICompatibleProvider()
    evaluator = FaithfulnessEvaluator(provider.get_judge_llm())
    report = evaluator.evaluate_file(dataset_path or None, report_path or None)
    typer.echo(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


@cli.command("migrate-conversations")
def migrate_conversations(source_sqlite_path: str):
    init_db()
    source_path = Path(source_sqlite_path)
    if not source_path.exists():
        raise typer.BadParameter(f"Source sqlite file not found: {source_path}")

    settings = get_settings()
    if settings.database_url.startswith("sqlite"):
        raise typer.BadParameter("Current DATABASE_URL is still SQLite; migrate target should be PostgreSQL.")

    source = sqlite3.connect(str(source_path))
    source.row_factory = sqlite3.Row
    db = SessionLocal()
    try:
        tables = {
            row["name"]
            for row in source.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "conversations" not in tables:
            typer.echo("No conversations table found in source SQLite; nothing migrated.")
            return

        rows = source.execute(
            """
            SELECT username, question, answer, grounded, sources_json, created_at
            FROM conversations
            ORDER BY id ASC
            """
        ).fetchall()
        imported = 0
        for row in rows:
            created_at = row["created_at"]
            if isinstance(created_at, str) and created_at:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            record = ConversationRecord(
                username=row["username"],
                question=row["question"],
                answer=row["answer"],
                grounded=bool(row["grounded"]),
                sources_json=row["sources_json"] or "[]",
                created_at=created_at,
            )
            db.add(record)
            imported += 1

        db.commit()
        typer.echo(f"Migrated {imported} conversations into {settings.database_url}")
    finally:
        source.close()
        db.close()


if __name__ == "__main__":
    cli()
