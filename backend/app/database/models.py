from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Department(Base):
    __tablename__ = "departments"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True)

    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)

    email = Column(String(255), unique=True)

    department_id = Column(
        Integer,
        ForeignKey("departments.id"),
        nullable=False
    )


class EmployeeRecord(Base):
    __tablename__ = "employee_records"

    id = Column(Integer, primary_key=True)

    employee_id = Column(
        Integer,
        ForeignKey("employees.id"),
        nullable=False
    )

    exam_score = Column(Integer, nullable=False)
    years_exp = Column(Integer, nullable=False)
    salary = Column(Float, nullable=False)