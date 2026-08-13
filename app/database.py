# ------------------------------------------------------------------
# Database connection setup and ORM model for logging predictions.
#
# Credentials are loaded from environment variables (.env file),
# never hardcoded, so they can be safely kept out of version control.
# ------------------------------------------------------------------

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime

# Load variables from the .env file into the environment
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Prediction(Base):
    """
    Stores every prediction request made through the API, along with
    the model's output. This acts as a simple prediction log / audit
    trail, a basic form of model monitoring.
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    smiles = Column(String, nullable=False)
    fda_approved_prob = Column(Float)
    ct_tox_prob = Column(Float)
    aromatic_amine_alert = Column(Boolean)
    predicted_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create all tables if they don't already exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that provides a database session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
