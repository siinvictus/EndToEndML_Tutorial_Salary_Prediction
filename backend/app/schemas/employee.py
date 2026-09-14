from typing import Optional

from pydantic import BaseModel, ConfigDict

# GET SCHEMAS

class DepartmentSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str


class EmployeeSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    first_name: str
    last_name: str
    email: Optional[str] = None
    department_id: int


class EmployeeRecordSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    employee_id: int
    exam_score: int
    years_exp: int
    salary: float

# POST SCHEMAS - note that we don't have id fields here because they are created automatically

class DepartmentCreate(BaseModel):
    name: str

class EmployeeCreate(BaseModel):
    first_name: str
    last_name: str
    email: Optional[str] = None
    department_id: Optional[int] = None

class EmployeeRecordCreate(BaseModel):
    employee_id: int
    exam_score: int
    years_exp: int
    salary: float

class PredictionSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    exam_score: int
    years_exp: int
    predicted_salary: float


class PredictionCreate(BaseModel):
    exam_score: int
    years_exp: int
