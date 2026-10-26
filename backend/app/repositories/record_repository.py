from sqlalchemy.orm import Session

from app.database.models import EmployeeRecord


def list_records(db: Session):
    return db.query(EmployeeRecord).all()


def create_record(db: Session, *, employee_id: int, exam_score: int,
                  years_exp: int, salary: float):
    db_record = EmployeeRecord(
        employee_id=employee_id,
        exam_score=exam_score,
        years_exp=years_exp,
        salary=salary,
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record
