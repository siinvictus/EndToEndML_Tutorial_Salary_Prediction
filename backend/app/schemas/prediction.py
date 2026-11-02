from pydantic import BaseModel, ConfigDict


class PredictionSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    exam_score: int
    years_exp: int
    predicted_salary: float


class PredictionCreate(BaseModel):
    exam_score: int
    years_exp: int


class PredictionContributionSchema(BaseModel):
    feature: str
    feature_value: float
    shap_value: float
    absolute_shap_value: float


class PredictionExplanationSchema(BaseModel):
    prediction_id: int
    predicted_salary: float
    base_value: float
    contributions: list[PredictionContributionSchema]