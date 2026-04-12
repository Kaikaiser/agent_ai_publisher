from typing import Optional

from sqlalchemy.orm import Session

from app.core.security import create_access_token, hash_password, verify_password
from app.db.models import User


class AuthService:
    def __init__(self, db: Session):
        self.db = db

    def authenticate(self, username: str, password: str) -> Optional[User]:
        user = self.db.query(User).filter(User.username == username).first()
        if not user or not user.is_active:
            return None
        if not verify_password(password, user.password_hash):
            return None
        return user

    def login(self, username: str, password: str) -> dict:
        user = self.authenticate(username, password)
        if not user:
            raise ValueError("账号或密码错误。")
        token = create_access_token(user.username)
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {"username": user.username, "role": user.role},
        }

    def create_admin(self, username: str, password: str) -> User:
        existing = self.db.query(User).filter(User.username == username).first()
        if existing:
            raise ValueError("用户名已存在。")
        user = User(username=username, password_hash=hash_password(password), role="admin", is_active=True)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user
