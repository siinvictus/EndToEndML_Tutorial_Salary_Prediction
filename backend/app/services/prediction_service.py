from sqlalchemy.orm import Session

from app.repositories import prediction_repository
from app.schemas.employee import PredictionCreate
from app.services import salary_prediction


class PredictionNotFoundError(LookupError):
    pass


class PredictionGenerationError(Exception):
    pass


class PredictionExplanationError(Exception):
    pass


def create_prediction(prediction_in: PredictionCreate, db: Session):
    try:
        predicted_salary = salary_prediction.predict_salary(
            exam_score=prediction_in.exam_score,
            years_exp=prediction_in.years_exp,
        )
    except (FileNotFoundError, ValueError) as error:
        raise PredictionGenerationError(str(error)) from error

    return prediction_repository.create_prediction(
        db,
        exam_score=prediction_in.exam_score,
        years_exp=prediction_in.years_exp,
        predicted_salary=predicted_salary,
    )


def explain_prediction(prediction_id: int, db: Session):
    prediction = prediction_repository.get_prediction(db, prediction_id)
    if prediction is None:
        raise PredictionNotFoundError("Prediction not found")

    try:
        explanation = salary_prediction.explain_salary_prediction(
            exam_score=prediction.exam_score,
            years_exp=prediction.years_exp,
            predicted_salary=prediction.predicted_salary,
        )
    except (FileNotFoundError, ValueError) as error:
        raise PredictionExplanationError(str(error)) from error

    return {"prediction_id": prediction.id, **explanation}
