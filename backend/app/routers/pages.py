from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.dependencies import get_db, templates
from app.services import department_service, employee_service, record_service


router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"title": "ML App Home"},
    )


@router.get("/departments", response_class=HTMLResponse)
def department_page(request: Request, db: Session = Depends(get_db)):
    db_departments = department_service.list_departments(db)
    return templates.TemplateResponse(
        request=request,
        name="departments.html",
        context={"departments": db_departments},
    )


@router.get("/employees", response_class=HTMLResponse)
def employee_page(request: Request, db: Session = Depends(get_db)):
    db_departments, db_employees = employee_service.get_employee_page_data(db)
    return templates.TemplateResponse(
        request=request,
        name="employees.html",
        context={"departments": db_departments, "employees": db_employees},
    )


@router.get("/records", response_class=HTMLResponse)
def records_page(request: Request, db: Session = Depends(get_db)):
    db_employees, db_records = record_service.get_record_page_data(db)
    return templates.TemplateResponse(
        request=request,
        name="records.html",
        context={"employees": db_employees, "records": db_records},
    )


@router.get("/predictions", response_class=HTMLResponse)
def predictions_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="predictions.html",
        context={},
    )


# Preserve the original duplicate route name and generated OpenAPI operation ID.
@router.get("/prediction_result", response_class=HTMLResponse, name="records_page")
def prediction_result_page(request: Request, db: Session = Depends(get_db)):
    db_employees, db_records = record_service.get_record_page_data(db)
    return templates.TemplateResponse(
        request=request,
        name="prediction_result.html",
        context={"employees": db_employees, "records": db_records},
    )
