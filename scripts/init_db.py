from app.database import SessionLocal, engine
from app import models

def init_db():
    models.Base.metadata.create_all(bind=engine)
    print("Database initialized successfully")

if __name__ == "__main__":
    init_db()
