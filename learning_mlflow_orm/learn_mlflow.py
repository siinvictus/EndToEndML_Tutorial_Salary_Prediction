import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mlflow 

mlflow.set_tracking_uri("http://127.0.0.1:5000")
# import and rename and drop nulls

data = pd.read_excel('data/salary_data.xlsx')
print(data.head(10))

data = data.rename(columns={'#': 'index', 'Exam Score (0–100)': 'exam_score', 
                                       'Years of Experience': 'years_exp', 'Salary (€)': 'salary'})

data = data.dropna()
# ----------------------------
# X / y SPLIT
# ----------------------------
X = data[['years_exp', 'exam_score']]
y = data['salary']

# ----------------------------
# TRAIN/TEST SPLIT
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# SCALING
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# MLFLOW RUN
# ----------------------------
with mlflow.start_run(run_name="test_ridge"):
    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    mlflow.log_param("alpha", 1.0)
    mlflow.log_param("model_type", "Ridge")
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("R2", r2)
    mlflow.sklearn.log_model(model, "model")

    print(f"Ridge — MAE: {mae:.2f}, R²: {r2:.4f}")