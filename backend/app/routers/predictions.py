from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.schemas.employee import PredictionCreate, PredictionExplanationSchema, PredictionSchema
from app.services import prediction_service


router = APIRouter()


@router.post("/predictions", response_model=PredictionSchema)
def create_prediction(prediction_in: PredictionCreate, db: Session = Depends(get_db)):
    try:
        return prediction_service.create_prediction(prediction_in, db)
    except prediction_service.PredictionGenerationError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate salary prediction: {error}",
        ) from error


@router.post("/predictions/{prediction_id}/explain", response_model=PredictionExplanationSchema)
def explain_prediction(prediction_id: int, db: Session = Depends(get_db)):
    try:
        return prediction_service.explain_prediction(prediction_id, db)
    except prediction_service.PredictionNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except prediction_service.PredictionExplanationError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not explain salary prediction: {error}",
        ) from error
