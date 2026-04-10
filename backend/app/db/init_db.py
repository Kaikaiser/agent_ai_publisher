import logging
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import hash_password
from app.db.base import Base
from app.db.models import User
from app.db.session import SessionLocal, engine

logger = logging.getLogger(__name__)


def init_db() -> None:
    settings = get_settings()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.faithfulness_dataset_path).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.faithfulness_report_path).parent.mkdir(parents=True, exist_ok=True)
    if settings.database_url.startswith("postgresql"):
        with engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)


def bootstrap_admin() -> None:
    settings = get_settings()
    db: Session = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == settings.default_admin_username).first()
        if existing:
            return
        admin = User(
            username=settings.default_admin_username,
            password_hash=hash_password(settings.default_admin_password),
            role="admin",
            is_active=True,
        )
        db.add(admin)
        db.commit()
        logger.info("Bootstrapped default admin user: %s", settings.default_admin_username)
    finally:
        db.close()
