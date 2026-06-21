import numpy as np
import pandas as pd

from src.predictor import Predictor
from src.trainer import Trainer


def test_predictor_returns_predictions_and_metrics():
    X_train = pd.DataFrame({"years_exp": [1, 2, 3, 4]})
    y_train = pd.Series([10, 20, 30, 40])
    model = Trainer(model_name="linear").train(X_train, y_train)
    predictor = Predictor(model)

    predictions = predictor.predict(pd.DataFrame({"years_exp": [5, 6]}))
    metrics = predictor.evaluate(np.array([50, 60]), predictions)

    assert np.allclose(predictions, [50, 60])
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
