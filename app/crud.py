from sqlalchemy.orm import Session
import numpy as np
from . import models, schemas
from .face_services import compare_faces

def create_user(db: Session, name: str, face_encoding):
    """ایجاد کاربر جدید در دیتابیس"""
    db_user = models.User(name=name)
    db_user.encoding_array = face_encoding
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_users(db: Session, skip: int = 0, limit: int = 100):
    """دریافت لیست کاربران"""
    return db.query(models.User).offset(skip).limit(limit).all()

def find_similar_face(db: Session, unknown_encoding, tolerance=0.6):
    """پیدا کردن شبیه‌ترین چهره در دیتابیس"""
    users = db.query(models.User).all()
    
    best_match = None
    best_distance = float('inf')
    
    for user in users:
        is_match, distance = compare_faces(user.encoding_array, unknown_encoding, tolerance)
        if is_match and distance < best_distance:
            best_match = user
            best_distance = distance
    
    if best_match:
        best_match.confidence = 1 - best_distance  # اضافه کردن confidence به نتیجه
        return best_match
    
    return None
