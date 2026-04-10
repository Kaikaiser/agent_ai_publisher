from app.auth.service import AuthService
from app.db.session import SessionLocal


def test_auth_service_login_success():
    db = SessionLocal()
    try:
        service = AuthService(db)
        payload = service.login('admin', 'admin123456')
        assert payload['user']['role'] == 'admin'
        assert payload['access_token']
    finally:
        db.close()
