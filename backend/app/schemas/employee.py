from pydantic import BaseModel, ConfigDict

class EmployeeRecordSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    employee_id: int
    exam_score: int
    years_exp: int
    salary: float

