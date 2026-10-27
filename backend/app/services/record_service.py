from sqlalchemy.orm import Session

from app.repositories import employee_repository, record_repository
from app.schemas.employee import EmployeeRecordCreate


class EmployeeNotFoundError(LookupError):
    pass


def get_record_page_data(db: Session):
    employees = employee_repository.list_employees(db)
    records = record_repository.list_records(db)
    return employees, records


def create_record(record_in: EmployeeRecordCreate, db: Session):
    employee = employee_repository.get_employee(db, record_in.employee_id)
    if employee is None:
        raise EmployeeNotFoundError("employee_id does not exist")

    return record_repository.create_record(
        db,
        employee_id=record_in.employee_id,
        exam_score=record_in.exam_score,
        years_exp=record_in.years_exp,
        salary=record_in.salary,
    )
