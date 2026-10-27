from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.schemas.employee import EmployeeRecordCreate, EmployeeRecordSchema
from app.services import record_service


router = APIRouter()


@router.post("/records", response_model=EmployeeRecordSchema)
def create_record(record_in: EmployeeRecordCreate, db: Session = Depends(get_db)):
    try:
        return record_service.create_record(record_in, db)
    except record_service.EmployeeNotFoundError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
