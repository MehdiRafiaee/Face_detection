import face_recognition
import numpy as np
from PIL import Image
import io
import cv2

async def process_uploaded_face(photo):
    """پردازش عکس آپلود شده و استخراج ویژگی‌های چهره"""
    contents = await photo.read()
    
    # تبدیل به numpy array
    image = np.array(Image.open(io.BytesIO(contents)))
    
    # تبدیل به RGB (اگر BGR باشد)
    if image.shape[2] == 3:
        image = image[:, :, ::-1]
    
    # تشخیص چهره‌ها
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    if not face_encodings:
        raise ValueError("No face detected in the image")
    
    if len(face_encodings) > 1:
        raise ValueError("Multiple faces detected. Please upload an image with only one face")
    
    return face_encodings[0]

def compare_faces(known_encoding, unknown_encoding, tolerance=0.6):
    """مقایسه دو چهره و برگرداندن شباهت"""
    distance = np.linalg.norm(known_encoding - unknown_encoding)
    return distance <= tolerance, distance
