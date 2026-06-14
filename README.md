# The Basics of Regression in Python

The following notebook is part of a training delivered through the **PyLadies Prishtina** community, covering the fundamentals of regression tasks through Simple Linear Regression, Multiple Linear Regression, Ridge, Lasso as well as Elastic Net. The dataset is simple and therefore the models are simple but the core idea is to show the full pipeline with the proper coding steps and stages, including but not limited to: the exploration in notebooks, the proper folder structure, keeping track of experiements, testing and deployment.

**Expected level of participants:** Beginners with some core high-school level mathematics and basic Python skills.

---

## Dataset

For the purposes of this work, we use the following toy dataset:

| Variable | Description |
|---|---|
| `exam_score` | Score on a professional test (0–100) |
| `years_exp` | Years of work experience |
| `salary` | Monthly salary in € *(target variable)* |

Our goal is to **predict salary** based on the two input variables. We experiment with different combinations of features and models to find the best approach.

---

## What This Project Covers

This project demonstrates a **full Data Science / Machine Learning pipeline** — not only preprocessing and model creation, but the complete path from raw data to a live application:

2.  **Exploratory Data Analysis** — distributions, correlations, missing value handling
3. **Preprocessing** — cleaning, feature selection, train/test split, scaling
4. **Modelling** — Simple Linear Regression (×2), Multiple Linear Regression, Ridge, Lasso, Elastic Net
5. **Hyperparamter Tuning** - Using both Optuna and the older GridSearch to show the differences among the two
6. **Evaluation** — models are compared using $R^2$ and RMSE (Root Mean Squared Error)
7. **Interpretation** - model predictions are interpreted using the SHAP method.
8. **Saving the model** — serialising the final trained model as a `.pkl` file
9. **Deployment** — serving the model via an API
10. **Web application** — a simple website where users can input values and get a salary prediction
11. **User testing** — testing the model with real users through the website

---

### How to go by it's structure:

1. **Marimo Notebook**: first explorations done here.
2. **The full Python-folder structure**: best-performing models are transformed into propoer project-structure in the **src** folder, reached through running **main.py**.`MLflow` tracking is integrated here to log each run.
3. **Models saved**: after running main, one should see the models saved as `.pkl` files in the **models** folder.
4. **Tests**: the **tests** folder contains unit tests for the pipeline, runnable via `pytest`.
5. 
6. **Reporting**: read the report to understand the full project and pipeline.

---

## Performance Metrics

| Metric | What it measures |
|---|---|
| $R^2$ | Proportion of variance in salary explained by the model (0 = useless, 1 = perfect) |
| RMSE | Average prediction error in euros — lower is better |

---

## Contributors

| Name | Role |
|---|---|
| Silva Bashllari| Author and Contributor|
| Deshira Randobrava | Contributor |
| Behare Konjuvca | Contributor |

---

## Community

This training was organised under **PyLadies Prishtina** — a community dedicated to supporting [not-only] women and other under-represented groups in Python and data science.
