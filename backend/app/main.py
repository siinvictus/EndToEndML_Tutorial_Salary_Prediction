from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database.connection import SessionLocal
from app.database.models import EmployeeRecord
from app.schemas.employee import EmployeeRecordSchema

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
def show_records(request: Request, db: Session = Depends(get_db)):
    db_records = db.query(EmployeeRecord).all()
    serialized_records = [EmployeeRecordSchema.model_validate(rec) for rec in db_records]

    return templates.TemplateResponse(request = request, name = "index.html", context = {"records": serialized_records})
    