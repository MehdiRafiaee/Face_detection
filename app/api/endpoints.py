from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from sqlalchemy.orm import Session
from typing import List
import os

from app.database import get_db
from app import schemas, crud
from app.services.face_service import process_uploaded_face, save_uploaded_file
from app.services.security import get_current_user

router = APIRouter()

@router.post("/register/", response_model=schemas.User)
async def register_user(
    name: str = Form(...),
    email: str = Form(None),
    photo: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """ثبت کاربر جدید با عکس"""
    try:
        if not photo.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # پردازش عکس و استخراج ویژگی‌های چهره
        face_encoding = await process_uploaded_face(photo)
        
        # ذخیره فایل آپلود شده
        upload_dir = os.getenv("UPLOAD_DIR", "uploads")
        file_path = save_uploaded_file(photo, upload_dir)
        
        # ایجاد کاربر در دیتابیس
        user_data = schemas.UserCreate(name=name, email=email)
        db_user = crud.create_user(db, user_data, face_encoding)
        
        return db_user
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/recognize/", response_model=schemas.FaceRecognitionResult)
async def recognize_face(
    photo: UploadFile = File(...),
    threshold: float = Form(0.6),
    db: Session = Depends(get_db)
):
    """تشخیص چهره در عکس ارسالی"""
    try:
        if not photo.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # پردازش عکس ارسالی
        face_encoding = await process_uploaded_face(photo)
        
        # مقایسه با چهره‌های موجود
        user = crud.find_similar_face(db, face_encoding, threshold)
        
        if user:
            return {
                "user_id": user.id,
                "name": user.name,
                "confidence": user.confidence,
                "message": "Face recognized successfully"
            }
        else:
            return {
                "message": "No matching face found"
            }
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/users/", response_model=List[schemas.User])
def get_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """دریافت لیست کاربران (نیاز به احراز هویت)"""
    users = crud.get_users(db, skip=skip, limit=limit)
    return users

@router.delete("/users/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: str = Depends(get_current_user)
):
    """حذف کاربر (نیاز به احراز هویت)"""
    success = crud.delete_user(db, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted successfully"}
