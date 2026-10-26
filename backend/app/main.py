from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.dependencies import get_db, templates
from app.routers import departments, employees, pages, predictions, records


app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Retain the original registration order, names, and generated operation IDs.
app.include_router(pages.router)
app.include_router(departments.router)
app.include_router(employees.router)
app.include_router(records.router)
app.include_router(predictions.router)
