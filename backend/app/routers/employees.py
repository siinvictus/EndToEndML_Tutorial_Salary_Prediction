from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.schemas.employee import EmployeeCreate, EmployeeSchema
from app.services import employee_service


router = APIRouter()


@router.post("/employees", response_model=EmployeeSchema)
def create_employee(employee_in: EmployeeCreate, db: Session = Depends(get_db)):
    try:
        return employee_service.create_employee(employee_in, db)
    except employee_service.DepartmentNotFoundError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
