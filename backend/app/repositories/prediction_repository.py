from sqlalchemy.orm import Session

from app.database.models import Prediction


def get_prediction(db: Session, prediction_id: int):
    return db.get(Prediction, prediction_id)


def create_prediction(db: Session, *, exam_score: int, years_exp: int,
                      predicted_salary: float):
    prediction = Prediction(
        exam_score=exam_score,
        years_exp=years_exp,
        predicted_salary=predicted_salary,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction
