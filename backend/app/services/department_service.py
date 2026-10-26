from sqlalchemy.orm import Session

from app.repositories import department_repository
from app.schemas.employee import DepartmentCreate


def list_departments(db: Session):
    return department_repository.list_departments(db)


def create_department(department_in: DepartmentCreate, db: Session):
    return department_repository.create_department(db, name=department_in.name)
