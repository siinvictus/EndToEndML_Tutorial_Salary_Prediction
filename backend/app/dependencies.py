from fastapi.templating import Jinja2Templates

from app.database.connection import SessionLocal


templates = Jinja2Templates(directory="app/templates")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
