from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import auth, chat, conversations, knowledge
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.init_db import bootstrap_admin, init_db

settings = get_settings()
configure_logging()

app = FastAPI(title="Publisher AI Assistant", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.backend_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(knowledge.router, prefix="/api/knowledge", tags=["knowledge"])
app.include_router(conversations.router, prefix="/api/conversations", tags=["conversations"])


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    bootstrap_admin()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
