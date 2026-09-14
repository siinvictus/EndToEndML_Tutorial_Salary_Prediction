from operator import call

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
from sqlalchemy.orm import Session
from fastapi.staticfiles import StaticFiles

from app.database.connection import SessionLocal
from app.database.models import Department, Employee, EmployeeRecord, Prediction
from app.schemas.employee import (
    DepartmentCreate,
    DepartmentSchema,
    EmployeeCreate,
    EmployeeSchema,
    EmployeeRecordSchema,
    EmployeeRecordCreate,
    PredictionSchema,
    PredictionCreate,
    PredictionExplanationSchema,
)
from app.services.salary_prediction import explain_salary_prediction, predict_salary

# ENDPOINTS IN MAIN.PY BUT USUALLY WHEN THE PROJECT IS BIGGER WE SEPARATE THEM IN ROUTER 

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
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

@app.get("/predictions", response_class=HTMLResponse)
def predictions_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="predictions.html",
        context={},
    )

# get by id based on prediction that was just inserted
@app.get("/prediction_result", response_class=HTMLResponse)
def records_page(request: Request, db: Session = Depends(get_db)):
    db_employees = db.query(Employee).all()
    db_records = db.query(EmployeeRecord).all()
    return templates.TemplateResponse(
        request=request,
        name="prediction_result.html",
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

@app.post("/predictions", response_model=PredictionSchema)
def create_prediction(
    prediction_in: PredictionCreate,
    db: Session = Depends(get_db)
):
    try:
        predicted_salary = predict_salary(
            exam_score=prediction_in.exam_score,
            years_exp=prediction_in.years_exp,
        )
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate salary prediction: {error}",
        ) from error

    prediction = Prediction(
        exam_score=prediction_in.exam_score,
        years_exp=prediction_in.years_exp,
        predicted_salary=predicted_salary,
    )

    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return prediction


@app.post("/predictions/{prediction_id}/explain", response_model=PredictionExplanationSchema)
def explain_prediction(prediction_id: int, db: Session = Depends(get_db)):
    prediction = db.get(Prediction, prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    try:
        explanation = explain_salary_prediction(
            exam_score=prediction.exam_score,
            years_exp=prediction.years_exp,
            predicted_salary=prediction.predicted_salary,
        )
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not explain salary prediction: {error}",
        ) from error

    return {
        "prediction_id": prediction.id,
        **explanation,
    }
