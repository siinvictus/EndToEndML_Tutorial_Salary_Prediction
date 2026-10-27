from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.schemas.employee import DepartmentCreate, DepartmentSchema
from app.services import department_service


router = APIRouter()


@router.post("/departments", response_model=DepartmentSchema)
def create_department(department_in: DepartmentCreate, db: Session = Depends(get_db)):
    return department_service.create_department(department_in, db)
