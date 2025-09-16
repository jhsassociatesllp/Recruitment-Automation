from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
from pymongo import MongoClient
from decouple import config
import jwt
from datetime import datetime, timedelta
from typing import Optional
import os
from database.db import users_collection
from dotenv import load_dotenv
from bson import ObjectId

load_dotenv()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 1 day

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Helper functions
def verify_password(plain_password, hashed_password):
    print(f"Verifying password: {plain_password}, hashed: {hashed_password}")  # Debug
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    print("In create access token")
    print(data)
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    print(f"Token expiration time: {expire}")
    to_encode.update({"exp": expire})
    print("Encode updated")
    print(SECRET_KEY)
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    print(encoded_jwt)
    return encoded_jwt

# async def get_current_user(token: str = Depends(oauth2_scheme)):
#     try:
#         print("In current user")
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         user_id = payload.get("sub")
#         print(f"Decoded user ID from token: {user_id}")  # Debug
#         if user_id is None:
#             raise HTTPException(status_code=401, detail="Invalid token")
#         user = users_collection.find_one({"_id": ObjectId(user_id)})
#         print(user)
#         if user is None:
#             raise HTTPException(status_code=401, detail="User not found")
#         return user
#     except jwt.ExpiredSignatureError:
#         raise HTTPException(status_code=401, detail="Token has expired")
#     except jwt.InvalidTokenError:
#         raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(token: str = Depends(oauth2_scheme)):
    print("In get_current_user")  # Debug
    try:
        print(f"Received token: {token}")  # Debug token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        print(f"Decoded user ID from token: {user_id}")  # Debug
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = users_collection.find_one({"_id": ObjectId(user_id)})
        print(f"Found user: {user}")  # Debug
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        print("Token has expired")  # Debug
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        print("Invalid token")  # Debug
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")  # Debug
        raise HTTPException(status_code=401, detail="Could not validate credentials")