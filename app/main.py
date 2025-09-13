from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
import uvicorn

from .database import SessionLocal, engine
from . import models, schemas, crud, face_services

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Face Recognition API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/register/", response_model=schemas.User)
async def register_user(
    name: str, 
    photo: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    """ثبت کاربر جدید با عکس"""
    try:
        # ذخیره عکس و استخراج ویژگی‌های چهره
        face_encoding = await face_services.process_uploaded_face(photo)
        
        # ایجاد کاربر در دیتابیس
        db_user = crud.create_user(db, name, face_encoding)
        
        return db_user
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recognize/")
async def recognize_face(
    photo: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    """تشخیص چهره در عکس ارسالی"""
    try:
        # پردازش عکس ارسالی
        face_encoding = await face_services.process_uploaded_face(photo)
        
        # مقایسه با چهره‌های موجود
        user = crud.find_similar_face(db, face_encoding)
        
        if user:
            return {"user_id": user.id, "name": user.name, "confidence": user.confidence}
        else:
            return {"message": "No matching face found"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/users/", response_model=List[schemas.User])
def get_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """دریافت لیست کاربران"""
    users = crud.get_users(db, skip=skip, limit=limit)
    return users

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
