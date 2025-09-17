from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from backend.routes.upload_resume import upload_router
from backend.routes.match_resumes import match_router
from backend.routes.whatsapp_routes import whatsapp_router
from backend.routes.auth_routes import auth_router 
from backend.routes.all_candidates import candidate_router
from backend.database.auth import get_current_user
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define frontend path
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
print("Frontend path:", frontend_path)
if os.path.exists(frontend_path):
    print("Files in frontend:", os.listdir(frontend_path))
else:
    print(f"Frontend directory not found at: {frontend_path}")

# Mount static files for assets like images
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Include routers
app.include_router(upload_router)
app.include_router(match_router)
app.include_router(candidate_router)
app.include_router(whatsapp_router)
app.include_router(auth_router)

# Serve index.html with authentication check
@app.get("/", response_class=FileResponse)
async def read_root(token: str = Depends(get_current_user)):
    if not token:
        return RedirectResponse(url="/login")
    index_path = os.path.join(frontend_path, "index.html")
    if not os.path.exists(index_path):
        print(f"Index file not found at: {index_path}")
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path, media_type="text/html")

# Serve login.html
@app.get("/login", response_class=FileResponse)
async def login_page():
    login_path = os.path.join(frontend_path, "login.html")
    if not os.path.exists(login_path):
        print(f"Login file not found at: {login_path}")
        raise HTTPException(status_code=404, detail="login.html not found")
    return FileResponse(login_path, media_type="text/html")

# Serve register.html
@app.get("/register", response_class=FileResponse)
async def register_page():
    register_path = os.path.join(frontend_path, "register.html")
    if not os.path.exists(register_path):
        print(f"Register file not found at: {register_path}")
        raise HTTPException(status_code=404, detail="register.html not found")
    return FileResponse(register_path, media_type="text/html")