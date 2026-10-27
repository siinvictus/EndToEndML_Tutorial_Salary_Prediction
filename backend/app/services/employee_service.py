from sqlalchemy.orm import Session

from app.repositories import department_repository, employee_repository
from app.schemas.employee import EmployeeCreate


class DepartmentNotFoundError(LookupError):
    pass


def get_employee_page_data(db: Session):
    departments = department_repository.list_departments(db)
    employees = employee_repository.list_employees(db)
    return departments, employees


def create_employee(employee_in: EmployeeCreate, db: Session):
    if employee_in.department_id is not None:
        department = department_repository.get_department(db, employee_in.department_id)
        if department is None:
            raise DepartmentNotFoundError("department_id does not exist")
    else:
        department = department_repository.find_department_by_name(db, "General")
        if department is None:
            department = department_repository.create_pending_department(db, name="General")

    return employee_repository.create_employee(
        db,
        first_name=employee_in.first_name,
        last_name=employee_in.last_name,
        email=employee_in.email,
        department_id=department.id,
    )
