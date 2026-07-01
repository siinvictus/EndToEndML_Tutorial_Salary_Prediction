from pathlib import Path
import pandas as pd
import random

from backend.app.database.connection import SessionLocal
from backend.app.database.models import Department, Employee, EmployeeRecord


# ----------------------------
# LOAD EXCEL
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR.parent / "data" / "salary_data.xlsx"

df = pd.read_excel(file_path)

# ----------------------------
# CLEAN DATA - do not clean it manually here vec vendose the actual dataset edhe me null values
# ----------------------------

df.columns = ["index", "exam_score", "years_exp", "salary"]
df = df.drop(columns=["index"])

# handle missing values
df["years_exp"] = df["years_exp"].fillna(df["years_exp"].mean())

# convert numpy types → python floats (IMPORTANT FIX)
df["exam_score"] = df["exam_score"].astype(float)
df["years_exp"] = df["years_exp"].astype(float)
df["salary"] = df["salary"].astype(float)


# ----------------------------
# DB SESSION
# ----------------------------

db = SessionLocal()


try:
    # ----------------------------
    # CREATE DEPARTMENTS (avoid duplicates)
    # ----------------------------

    departments = ["Engineering", "HR", "Finance", "Marketing"]
    dept_objects = []

    for name in departments:
        existing = db.query(Department).filter_by(name=name).first()

        if existing:
            dept_objects.append(existing)
        else:
            dept = Department(name=name)
            db.add(dept)
            dept_objects.append(dept)

    db.commit()

    # refresh to get IDs
    for d in dept_objects:
        db.refresh(d)


    # ----------------------------
    # INSERT EMPLOYEES + RECORDS
    # ----------------------------

    for i, row in df.iterrows():

        dept = random.choice(dept_objects)

        employee = Employee(
            first_name=f"User{i}",
            last_name="Test",
            email=f"user{i}@test.com",
            department_id=dept.id
        )

        db.add(employee)
        db.flush()  # get employee.id before commit

        record = EmployeeRecord(
            employee_id=employee.id,
            exam_score=float(row["exam_score"]),
            years_exp=float(row["years_exp"]),
            salary=float(row["salary"])
        )

        db.add(record)

    db.commit()

    print("✅ Data imported successfully")


finally:
    db.close()