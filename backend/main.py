from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from routes.upload_resume import upload_router
from routes.match_resumes import match_router
from routes.whatsapp_routes import whatsapp_router
from routes.auth_routes import auth_router 
from routes.all_candidates import candidate_router
import os
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.include_router(upload_router)
app.include_router(match_router)
app.include_router(candidate_router)
app.include_router(whatsapp_router)
app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set this to your frontend domain instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend folder
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Serve index.html as root
@app.get("/")
def read_root():
    return FileResponse(os.path.join(frontend_path, "index.html"))

# Serve login.html
@app.get("/login")
def login_page():
    return FileResponse(os.path.join(frontend_path, "login.html"))

# Serve register.html
@app.get("/register")
def register_page():
    return FileResponse(os.path.join(frontend_path, "register.html"))