# The Basics of Regression in Python

The following notebook is part of a training delivered through the **PyLadies Prishtina** community, covering the fundamentals of regression tasks through Simple Linear Regression, Multiple Linear Regression, Ridge, Lasso as well as Elastic Net. The dataset is simple and therefore the models are simple but the core idea is to show the full pipeline with the proper coding steps and stages, including but not limited to: the exploration in notebooks, the proper folder structure, keeping track of experiements, testing and deployment.

**Expected level of participants:** Beginners with some core high-school level mathematics and basic Python skills.

<p align="center">
  <img src="extra_images/PyLadies_Prishtina_Logo.jpeg" width="200">
</p>


---

## Dataset

For the purposes of this work, we use the following toy dataset:

| Variable | Description |
|---|---|
| `exam_score` | Score on a professional test (0–100) |
| `years_exp` | Years of work experience |
| `salary` | Monthly salary in € *(target variable)* |

Our goal is to **predict salary** based on the two input variables. We experiment with different combinations of features and models to find the best approach.

## Performance Metrics

| Metric | What it measures |
|---|---|
| $R^2$ | Proportion of variance in salary explained by the model (0 = useless, 1 = perfect) |
| RMSE | Average prediction error in euros — lower is better |

---

## What This Project Covers

This project demonstrates a **full Data Science / Machine Learning pipeline** — not only preprocessing and model creation, but the complete path from raw data to a live application:
1. **Exploratory Data Analysis** — distributions, correlations, missing value handling
2. **Preprocessing** — cleaning, feature selection, train/test split, scaling
3. **Modelling** — Simple Linear Regression (×2), Multiple Linear Regression, Ridge, Lasso, Elastic Net
4. **Hyperparamter Tuning** - Using both Optuna and the older GridSearch to show the differences among the two
5. **Evaluation** — models are compared using $R^2$ and RMSE (Root Mean Squared Error)
6. **Interpretation** - model predictions are interpreted using the SHAP method.
7. **Saving the model** — serialising the final trained model as a `.pkl` file

   
[Upcoming part]


9. **Deployment** — serving the model via an FASTAPI
10. **Web application** — a simple website where users can input values and get a salary prediction
11. **User testing** — testing the model with test classes
12. **Simple Database** -- a simple experiment to extract from a database the CSV file needed for the model training as well as saving the user data in the DB using Postgres, SQLAlchemy, Alembic (for the migration), PyDantic

---

### Base Installations:

1. Install **Visual Studio Code**. <br>
2. Install the **VS Code Python extension**. <br>
3. Install **Git** so you can clone/download the project on the new PC. <br>
4. Install **Python 3.12** and make sure it is added to PATH. Check it with `python --version`. <br>
5. Install **uv** for project environment and dependency management. On Windows, you can install it with `winget install --id=astral-sh.uv -e`. <br>
6. Install **PostgreSQL** if you want to run the database/ORM parts of the project. You can also install **pgAdmin** or the **PostgreSQL VS Code extension** to inspect the database visually. <br>
7. Open the project in VS Code, then use the terminal to run `uv sync` so uv creates the virtual environment and installs the project dependencies from `pyproject.toml` and `uv.lock`. <br>

---

### How to Study/ Go by this project:
1. Understand the core concepts of data analysis using the **Marimo Notebook**, first explorations are done here. Specificially, go to the folder  `notebooks` and you will see there `experientation_notebook`. In your command line, just by going to the folder write  `marimo edit experientation_notebook` and follow the path on your local pc.
2. Before starting with production code, get comfortable with a small test of learning orm in python and ml flow using the marimo edit `learning_materials` folder.
3. the main starts with the sqlalchemy part that creates a db and extracts the data we need [This is upcoming]
4. Then that data is stored in the data folder [You can already find the data there]
5. **The full Python-folder structure**: best-performing models are transformed into propper project-structure in the **src** folder, reached through running **main.py**.
## Usage

Run the pipeline from the project root:

```bash
python main.py --data path/to/data.csv
```

### Arguments

| Flag | Required | Default | Description |
|---|---|---|---|
| `--data` | ✅ | — | Path to a CSV or Excel dataset |
| `--model` | ❌ | `linear` | Model to train: `linear` or `lasso` |
| `--alpha` | ❌ | `1.0` | Lasso alpha value (only used with `--model lasso`) |
| `--scaling` | ❌ | `standard` | Feature scaling: `standard`, `minmax`, or `none` |
| `--test-size` | ❌ | `0.2` | Proportion of data used for testing |
| `--random-state` | ❌ | `42` | Random seed for reproducible train/test splits |
| `--model-output` | ❌ | `models/<model>_regression.pkl` | Custom path to save the trained model |
| `--exam-score` | ❌ | off | If you want to predict a specific person's salary, you can add their exam score |
| `--years-exp` | ❌ | off | If you want to predict a specific person's salary, you can add their years of experience |
| `--explain` | ❌ | off | Print SHAP feature importance for the above person |

*Note: If you want to predict the salary for someone, you must add <b> both </b> the years of experience and salary. 
       If you want to explain someone's predicted salary you must have had the above two first.

### Examples

Train a linear regression model with default settings:
```bash
python main.py --data data/salaries.csv
```

Train a lasso model with a custom alpha and minmax scaling:
```bash
python main.py --data data/salaries.csv --model lasso --alpha 0.5 --scaling minmax
```

Train, predict and print SHAP explanations for a person whose years of experience are 3 and exam socre is 80:
```bash
python main.py --data data/salaries.csv --years-exp 3.0 --exam-score 80 --explain
```

Save the model to a custom path:
```bash
python main.py --data data/salaries.csv --model-output models/my_model.pkl
```

---

## Folder Structure
```text
├── notebooks/ 
│   └── salary_regression.py        # EXPLORATION ONLY
│       (marimo: EDA, trying models, plots,
│        SHAP exploration, scratch work)
│       → nothing here is "production", 
│         it's where YOU learn and decide 
│         what the final pipeline should do
│
|── learning_materials/
│   └── learn_mlflow.py             # learn mlflow
│   └── test_models                 # make db schema and connection to db
│   └── test_app                    # query db
|   └── other_python_stuff          # learn about general concepts like data classes, error handling etc.  
├── data/
│   └── salary_data.xlsx             # raw source data
│
├── src/
│   ├── data_loader.py               # loads raw data
│   ├── preprocessor.py              # cleans, scales, splits
│   ├── trainer.py                   # trains models, 
│   │                                  logs to mlflow
│   ├── predictor.py                 # loads trained model,
│   │                                  predicts, explains (SHAP)
│   └── database/                    # NEW — where DB fits  [Upcoming]
│       ├── connection.py            # engine, SessionLocal
│       ├── models.py                # Department, Employee,
│       │                              EmployeeRecord tables
│       └── crud.py                  # reusable DB operations
│
├── migrations/                      # Alembic schema versions [Upcoming]
│
├── models/
│   └── best_model.pkl               # saved trained model
│
├── scripts/                         # [Upcoming]
│   ├── seed_database.py             # one-time: populate DB
│   │                                  with dummy data
│   └── train_and_log.py             # one-time/repeatable:
│                                       run training, log to mlflow
│
├── api/                             # [Upcoming]
│   └── main.py                      # FastAPI app — the 
│                                       "front door" tying 
│                                       everything together
│
├── tests/
│   └── test_data_loader.py         # pytest
│   └── test_predictor.py            # pytest
│   └── test_preprocessor.py         # pytest
│   └── test_trainer.py              # pytest
|
|
│
└── main.py                          # simple CLI entrypoint
```

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
