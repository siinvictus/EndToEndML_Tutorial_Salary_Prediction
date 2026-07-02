from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database.connection import SessionLocal
from app.database.models import Department, Employee, EmployeeRecord
from app.schemas.employee import (
    DepartmentCreate,
    DepartmentSchema,
    EmployeeCreate,
    EmployeeSchema,
    EmployeeRecordSchema,
    EmployeeRecordCreate,
)

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "ML App Home"},
    )


@app.get("/departments", response_class=HTMLResponse)
def department_page(request: Request, db: Session = Depends(get_db)):
    db_departments = db.query(Department).all()
    return templates.TemplateResponse(
        request=request,
        name="departments.html",
        context={"departments": db_departments},
    )


@app.get("/employees", response_class=HTMLResponse)
def employee_page(request: Request, db: Session = Depends(get_db)):
    db_departments = db.query(Department).all()
    db_employees = db.query(Employee).all()
    return templates.TemplateResponse(
        request=request,
        name="employees.html",
        context={
            "departments": db_departments,
            "employees": db_employees,
        },
    )


@app.get("/records", response_class=HTMLResponse)
def records_page(request: Request, db: Session = Depends(get_db)):
    db_employees = db.query(Employee).all()
    db_records = db.query(EmployeeRecord).all()
    return templates.TemplateResponse(
        request=request,
        name="records.html",
        context={
            "employees": db_employees,
            "records": db_records,
        },
    )


@app.post("/departments", response_model=DepartmentSchema)
def create_department(department_in: DepartmentCreate, db: Session = Depends(get_db)):
    department = Department(name=department_in.name)
    db.add(department)
    db.commit()
    db.refresh(department)
    return department


@app.post("/employees", response_model=EmployeeSchema)
def create_employee(employee_in: EmployeeCreate, db: Session = Depends(get_db)):
    department = None
    if employee_in.department_id is not None:
        department = db.get(Department, employee_in.department_id)
        if department is None:
            raise HTTPException(status_code=400, detail="department_id does not exist")
    else:
        department = db.query(Department).filter_by(name="General").first()
        if department is None:
            department = Department(name="General")
            db.add(department)
            db.flush()

    employee = Employee(
        first_name=employee_in.first_name,
        last_name=employee_in.last_name,
        email=employee_in.email,
        department_id=department.id,
    )
    db.add(employee)
    db.commit()
    db.refresh(employee)
    return employee


@app.post("/records", response_model=EmployeeRecordSchema)
def create_record(record_in: EmployeeRecordCreate, db: Session = Depends(get_db)):
    employee = db.get(Employee, record_in.employee_id)
    if employee is None:
        raise HTTPException(status_code=400, detail="employee_id does not exist")

    db_record = EmployeeRecord(
        employee_id=record_in.employee_id,
        exam_score=record_in.exam_score,
        years_exp=record_in.years_exp,
        salary=record_in.salary,
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return db_record
    