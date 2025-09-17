from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from backend.routes.upload_resume import upload_router
from backend.routes.match_resumes import match_router
from backend.routes.whatsapp_routes import whatsapp_router
from backend.routes.auth_routes import auth_router 
from backend.routes.all_candidates import candidate_router
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = os.path.join(os.path.dirname(__file__), "frontend")  # Adjusted path
print("Frontend path:", frontend_path)
print("Files in frontend:", os.listdir(frontend_path))

app.include_router(upload_router)
app.include_router(match_router)
app.include_router(candidate_router)
app.include_router(whatsapp_router)
app.include_router(auth_router)

@app.get("/", response_class=FileResponse)
async def read_root():
    index_path = os.path.join(frontend_path, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path, media_type="text/html")

@app.get("/login", response_class=FileResponse)
async def login_page():
    login_path = os.path.join(frontend_path, "login.html")
    if not os.path.exists(login_path):
        raise HTTPException(status_code=404, detail="login.html not found")
    return FileResponse(login_path, media_type="text/html")

@app.get("/register", response_class=FileResponse)
async def register_page():
    register_path = os.path.join(frontend_path, "register.html")
    if not os.path.exists(register_path):
        raise HTTPException(status_code=404, detail="register.html not found")
    return FileResponse(register_path, media_type="text/html")