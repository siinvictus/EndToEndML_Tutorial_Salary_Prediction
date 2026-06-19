from sqlalchemy import create_engine, Column, Integer, String, Float
import os
from dotenv import load_dotenv
from sqlalchemy.orm import declarative_base

load_dotenv(".passwords.")
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

engine = create_engine(TEST_DATABASE_URL)

Base = declarative_base()

# STEP 1: define your table class(es) FIRST
class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    budget = Column(Float, nullable=True)
    #timeframe = Column(String, nullable = True) #now this new column gets added, what do we do? It won't show in the db because there is no version control, 
    # we need alembic

# STEP 2: THEN create the tables, after classes exist;
# this only checks if the table exists and if it doesn't it creates it
# it doesn't keep up to date the row changes and all
Base.metadata.create_all(engine)