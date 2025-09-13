import face_recognition
import numpy as np
from PIL import Image
import io
import cv2
from typing import Tuple, List
import os

async def process_uploaded_face(photo) -> np.ndarray:
    """پردازش عکس آپلود شده و استخراج ویژگی‌های چهره"""
    contents = await photo.read()
    
    try:
        image = np.array(Image.open(io.BytesIO(contents)))
        
        # تبدیل به RGB اگر لازم باشد
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image[:, :, ::-1]
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3][:, :, ::-1]
        
        # تشخیص چهره‌ها
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        if not face_encodings:
            raise ValueError("No face detected in the image")
        
        if len(face_encodings) > 1:
            raise ValueError("Multiple faces detected. Please upload an image with only one face")
        
        return face_encodings[0]
    
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def compare_faces(known_encoding: np.ndarray, unknown_encoding: np.ndarray, tolerance: float = 0.6) -> Tuple[bool, float]:
    """مقایسه دو چهره و برگرداندن شباهت"""
    if known_encoding.shape != unknown_encoding.shape:
        raise ValueError("Encodings must have the same shape")
    
    distance = np.linalg.norm(known_encoding - unknown_encoding)
    similarity = 1 - distance
    is_match = similarity >= tolerance
    
    return is_match, similarity

def save_uploaded_file(file, upload_dir: str) -> str:
    """ذخیره فایل آپلود شده"""
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        content = file.file.read()
        buffer.write(content)
    
    return file_path

def validate_image_size(image: np.ndarray, max_size: Tuple[int, int] = (2000, 2000)) -> bool:
    """اعتبارسنجی اندازه تصویر"""
    height, width = image.shape[:2]
    return height <= max_size[0] and width <= max_size[1]
