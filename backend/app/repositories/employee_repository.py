from sqlalchemy.orm import Session

from app.database.models import Employee


def list_employees(db: Session):
    return db.query(Employee).all()


def get_employee(db: Session, employee_id: int):
    return db.get(Employee, employee_id)


def create_employee(db: Session, *, first_name: str, last_name: str,
                    email: str | None, department_id: int):
    employee = Employee(
        first_name=first_name,
        last_name=last_name,
        email=email,
        department_id=department_id,
    )
    db.add(employee)
    db.commit()
    db.refresh(employee)
    return employee
