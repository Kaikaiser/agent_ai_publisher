from pathlib import Path

import typer

from app.auth.service import AuthService
from app.core.config import get_settings
from app.db.init_db import bootstrap_admin, init_db
from app.db.session import SessionLocal
from app.services.chat import ChatService
from app.services.knowledge import KnowledgeService
from app.services.providers import OpenAIProvider

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
    provider = OpenAIProvider()
    try:
        service = KnowledgeService(db, provider.get_embeddings())
        job = service.import_file(file, Path(file).name, username, book_title, doc_type, allowed_role)
        typer.echo(f"Import completed: job={job.id} message={job.message}")
    finally:
        db.close()


@cli.command("ask")
def ask(username: str, question: str, role: str = "admin", book_title: str = "", doc_type: str = ""):
    init_db()
    db = SessionLocal()
    provider = OpenAIProvider()
    try:
        service = ChatService(db, provider.get_llm(), provider.get_embeddings())
        result = service.ask(username, role, question, book_title or None, doc_type or None)
        typer.echo(result["answer"])
    finally:
        db.close()


if __name__ == "__main__":
    cli()
