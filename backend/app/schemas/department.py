from pydantic import BaseModel, ConfigDict


class DepartmentSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str


class DepartmentCreate(BaseModel):
    name: str