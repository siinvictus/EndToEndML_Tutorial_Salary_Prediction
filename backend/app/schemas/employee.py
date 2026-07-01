from pydantic import BaseModel

class EmployeeRecordSchema(BaseModel):
    id: int
    employee_id: int
    exam_score: int
    years_exp: int
    salary: float

    class Config:
        from_attributes = True  # Allows Pydantic to read directly from your SQLAlchemy object
