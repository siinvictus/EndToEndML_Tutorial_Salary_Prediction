from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://postgres:1234@localhost:5433/salary_prediction"

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# 🔍 QUICK CONNECTION TEST
try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        print("✅ Database connected successfully:", result.scalar())
except Exception as e:
    print("❌ Database connection failed:")
    print(e)
    