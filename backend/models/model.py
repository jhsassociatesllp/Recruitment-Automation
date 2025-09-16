from pydantic import BaseModel
from typing import List, Dict, Optional
from fastapi import Query

# Pydantic models
class JobDescription(BaseModel):
    role: str
    jd: str
    threshold: float | None

class InterviewRequest(BaseModel):
    time_slot: str
    mode: str  # "online" or "offline"

class Candidate(BaseModel):
    name: str
    phone: str
    email: str
    resume: str
    score: float

class InterviewSchedule(BaseModel):
    candidate_id: str
    time_slot: str
    mode: str
    
# Pydantic model for a single candidate
class Candidate(BaseModel):
    candidate_id: str
    candidate_name: str
    mobile: str
    email: str
    role: str
    hr_name: str

# Pydantic model for the request with multiple candidates
class MessageRequest(BaseModel):
    candidates: List[Candidate]
    mode: str
    slots: Dict[str, List[str]]
    
class RescheduleRequest(BaseModel):
    candidate_id: str
    mode: str
    slots: Dict[str, List[str]]
    
# Models
class UserCreate(BaseModel):
    name: str
    email: str
    role: str  # 'HR' or 'Partner'
    password: str
    confirm_password: str

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    name: str
    role: str
    access_token: str
    token_type: str
    
class MatchRequest(BaseModel):
    role: str
    jd: str
    threshold: float = 40.0
    
class CandidateFilter(BaseModel):
    search: Optional[str] = None
    age_group: Optional[str] = None
    experience: Optional[str] = None
    skills: Optional[List[str]] = Query(None)
    location: Optional[str] = None
    qualification: Optional[str] = None