from sqlalchemy import Column, Integer, String, LargeBinary, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import pickle
import numpy as np

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    email = Column(String, unique=True, index=True)
    face_encoding = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    confidence_threshold = Column(Float, default=0.6)

    @property
    def encoding_array(self):
        return pickle.loads(self.face_encoding)
    
    @encoding_array.setter
    def encoding_array(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("Encoding must be a numpy array")
        self.face_encoding = pickle.dumps(value)

class AdminUser(Base):
    __tablename__ = "admin_users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
