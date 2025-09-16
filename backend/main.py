from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.upload_resume import upload_router
from routes.match_resumes import match_router
from routes.whatsapp_routes import whatsapp_router
from routes.auth_routes import auth_router 
from routes.all_candidates import candidate_router

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

@app.get("/")
def home():
    return {"message": "Welcome to the Automated Recruitment System API"}