from typing import Optional

from pydantic import BaseModel, ConfigDict

class EmployeeSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    first_name: str
    last_name: str
    email: Optional[str] = None
    department_id: int


class EmployeeCreate(BaseModel):
    first_name: str
    last_name: str
    email: Optional[str] = None
    department_id: Optional[int] = None
