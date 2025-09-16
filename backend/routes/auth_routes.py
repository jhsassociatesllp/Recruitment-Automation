from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
from pymongo import MongoClient
from decouple import config
import jwt
from datetime import datetime, timedelta
from typing import Optional
from database.db import *
from database.auth import *
from models.model import *

# Router
auth_router = APIRouter()

@auth_router.post("/register/", status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate):
    if user.password != user.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    if users_collection.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(user.password)
    user_doc = {
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "password": hashed_password
    }
    result = users_collection.insert_one(user_doc)
    return {"message": "User registered successfully", "user_id": str(result.inserted_id)}

@auth_router.post("/login/", response_model=Token)
async def login(user: UserLogin):
    print(f"Received email: {user.email}, password: {user.password}")  # Debug
    user_doc = users_collection.find_one({"email": user.email})
    if not user_doc:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(user.password, user_doc["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    print("Password verified")
    user_id_str = str(user_doc["_id"])
    print(f"User ID: {user_id_str}")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    print("Access token expiers")
    print(user_doc)
    access_token = create_access_token(
        data={"sub": user_id_str, "role": user_doc["role"]}, expires_delta=access_token_expires
    )
    print(f"Access token: {access_token}")
    return {"name": user_doc["name"], "role": user_doc["role"], "access_token": access_token, "token_type": "bearer"}

@auth_router.get("/protected/")
async def protected_route(current_user: dict = Depends(get_current_user)):
    return {"message": "Protected route accessed", "user": {"name": current_user["name"], "role": current_user["role"]}}