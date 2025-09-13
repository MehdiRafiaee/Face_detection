import pytest
import numpy as np
from app.services.face_service import compare_faces

def test_face_comparison():
    """تست مقایسه چهره‌ها"""
    # ایجاد دو encoding مشابه
    encoding1 = np.random.rand(128)
    encoding2 = encoding1 + np.random.normal(0, 0.01, 128)
    
    is_match, similarity = compare_faces(encoding1, encoding2, tolerance=0.6)
    
    assert isinstance(is_match, bool)
    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1

def test_face_comparison_different():
    """تست مقایسه چهره‌های مختلف"""
    encoding1 = np.random.rand(128)
    encoding2 = np.random.rand(128)
    
    is_match, similarity = compare_faces(encoding1, encoding2, tolerance=0.6)
    
    assert not is_match
    assert similarity < 0.6
