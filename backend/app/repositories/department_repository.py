from sqlalchemy.orm import Session

from app.database.models import Department


def list_departments(db: Session):
    return db.query(Department).all()


def get_department(db: Session, department_id: int):
    return db.get(Department, department_id)


def find_department_by_name(db: Session, name: str):
    return db.query(Department).filter_by(name=name).first()


def create_department(db: Session, *, name: str):
    department = Department(name=name)
    db.add(department)
    db.commit()
    db.refresh(department)
    return department


def create_pending_department(db: Session, *, name: str):
    # The employee operation commits this department and employee together.
    department = Department(name=name)
    db.add(department)
    db.flush()
    return department
