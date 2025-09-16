from pymongo import MongoClient
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv  

load_dotenv()

# MongoDB setup
client = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
db = client["Recruitment_Automation"]
resumes_collection = db["resumes"]
matched_resumes_collection = db["matched_resumes"]
interviews_collection = db["interviews"]
messaged_collection = db["Messaged"]
scheduled_collection = db["Scheduled"]
users_collection = db["users"]