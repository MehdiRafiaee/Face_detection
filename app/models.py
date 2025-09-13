from sqlalchemy import Column, Integer, String, LargeBinary, Float
from sqlalchemy.ext.declarative import declarative_base
import numpy as np
import pickle

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    face_encoding = Column(LargeBinary)  # ذخیره encoding به صورت binary
    created_at = Column(Integer)  # timestamp

    @property
    def encoding_array(self):
        """تبدیل encoding باینری به آرایه numpy"""
        return pickle.loads(self.face_encoding)
    
    @encoding_array.setter
    def encoding_array(self, value):
        """تبدیل آرایه numpy به باینری برای ذخیره در دیتابیس"""
        self.face_encoding = pickle.dumps(value)
