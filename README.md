# The Basics of Regression in Python

The following notebook is part of a training delivered by **Silva Bashllari, MSc** through the **PyLadies Prishtina** community, covering the fundamentals of regression tasks through Simple Linear Regression, Multiple Linear Regression, Ridge, and Lasso.

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

1. **Exploratory Data Analysis** — distributions, correlations, missing value handling
2. **Preprocessing** — cleaning, feature selection, train/test split, scaling
3. **Modelling** — Simple Linear Regression (×2), Multiple Linear Regression, Ridge, Lasso
4. **Evaluation** — models are compared using $R^2$ and RMSE (Root Mean Squared Error)
5. **Saving the model** — serialising the final trained model as a `.pkl` file
6. **Deployment** — serving the model via an API
7. **Web application** — a simple website where users can input values and get a salary prediction
8. **User testing** — testing the model with real users through the website

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

This training was organised under **[PyLadies Prishtina](https://github.com/PyLadiesPrishtina)** — a community dedicated to supporting women and gender minorities in Python and data science.
