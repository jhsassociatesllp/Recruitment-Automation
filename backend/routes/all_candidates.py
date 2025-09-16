# from fastapi import APIRouter, Depends, HTTPException, Query
# from typing import Optional, List
# from bson import ObjectId
# import json
# from database.db import resumes_collection
# from database.auth import get_current_user

# candidate_router = APIRouter()

# @candidate_router.get("/candidates/")
# async def get_candidates(
#     search: Optional[str] = Query(None),
#     experience: Optional[str] = Query(None),
#     location: Optional[str] = Query(None),
#     qualification: Optional[str] = Query(None),
#     current_user: dict = Depends(get_current_user)
# ):
#     try:
#         pipeline = []

#         match_criteria = {}

#         if search:
#             match_criteria["$or"] = [
#                 {"data.full_name": {"$regex": search, "$options": "i"}},
#                 {"data.relevant_roles": {"$regex": search, "$options": "i"}},
#                 {"data.experience": {"$regex": search, "$options": "i"}},
#                 {"data.location": {"$regex": search, "$options": "i"}},
#                 {"data.highest_qualification": {"$regex": search, "$options": "i"}}
#             ]

#         if experience and experience in ["Fresher", "0–1 year", "1–3 years", "3–5 years", ">5 years"]:
#             match_criteria["data.experience_category"] = experience

#         if location:
#             match_criteria["data.LocationCategory"] = location

#         if qualification and qualification in ["Bachelor's", "Master's", "MBA", "PhD", "Diploma"]:
#             match_criteria["data.QualificationCategory"] = qualification

#         if match_criteria:
#             pipeline.append({"$match": match_criteria})

#         pipeline.append({
#             "$project": {
#                 "_id": 1,
#                 "candidate_id": 1,
#                 "name": {"$ifNull": ["$data.full_name", "N/A"]},
#                 "relevant_roles": "$data.relevant_roles",
#                 "experience_category": {"$ifNull": ["$data.experience_category", "Fresher"]},
#                 "experience_value": {
#                     "$cond": {
#                         "if": {"$eq": ["$data.experience_value", None]},
#                         "then": "0 years",
#                         "else": {
#                             "$concat": [
#                                 {"$toString": {"$trunc": {"$toDouble": {"$arrayElemAt": [{"$split": ["$data.experience_value", "."]}, 0]}}}},
#                                 " years"
#                             ]
#                         }
#                     }
#                 },
#                 "LocationCategory": {"$ifNull": ["$data.LocationCategory", "N/A"]},
#                 "LocationExact": {"$ifNull": ["$data.LocationExact", "N/A"]},
#                 "QualificationCategory": {"$ifNull": ["$data.QualificationCategory", "N/A"]},
#                 "QualificationExact": {"$ifNull": ["$data.QualificationExact", "N/A"]},
#                 "certifications": {
#                     "$cond": {
#                         "if": {"$eq": [{"$size": {"$ifNull": ["$data.certifications", []]}}, 0]},
#                         "then": "No",
#                         "else": "$data.certifications"
#                     }
#                 },
#                 "file_name": 1,
#                 "reference_name": 1
#             }
#         })

#         candidates = list(resumes_collection.aggregate(pipeline))
#         # Manually encode the data to handle ObjectId and other non-JSON-serializable types
#         def custom_encoder(obj):
#             if isinstance(obj, ObjectId):
#                 return str(obj)
#             raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

#         candidates_json = json.dumps(candidates, default=custom_encoder)

#         return {"candidates": json.loads(candidates_json)}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error fetching candidates: {str(e)}")

# @candidate_router.get("/locations")
# async def get_locations(current_user: dict = Depends(get_current_user)):
#     try:
#         pipeline = [
#             {"$group": {"_id": "$data.LocationCategory", "count": {"$sum": 1}}},
#             {"$match": {"_id": {"$ne": None}}},
#             {"$sort": {"_id": 1}}
#         ]
#         locations = [doc["_id"] for doc in resumes_collection.aggregate(pipeline) if doc["_id"] != "N/A"]
#         return {"locations": locations}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error fetching locations: {str(e)}")



from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List
from bson import ObjectId
import json
from database.db import resumes_collection
from database.auth import get_current_user

candidate_router = APIRouter()

@candidate_router.get("/candidates/")
async def get_candidates(
    search: Optional[str] = Query(None),
    experience: Optional[str] = Query(None),
    location: Optional[str] = Query(None),
    qualification: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    try:
        pipeline = []

        match_criteria = {}

        if search:
            match_criteria["$or"] = [
                {"data.full_name": {"$regex": search, "$options": "i"}},
                {"data.relevant_roles": {"$regex": search, "$options": "i"}},
                {"data.experience_category": {"$regex": search, "$options": "i"}},
                {"data.LocationCategory": {"$regex": search, "$options": "i"}},
                {"data.QualificationCategory": {"$regex": search, "$options": "i"}}
            ]

        if experience and experience in ["Fresher", "0–1 year", "1–3 years", "3–5 years", ">5 years"]:
            match_criteria["data.experience_category"] = experience

        if location:
            match_criteria["data.LocationCategory"] = location

        if qualification and qualification in ["Bachelor's", "Master's", "MBA", "PhD", "Diploma"]:
            match_criteria["data.QualificationCategory"] = qualification

        if match_criteria:
            pipeline.append({"$match": match_criteria})

        pipeline.append({
            "$project": {
                "_id": 1,
                "candidate_id": 1,
                "name": {"$ifNull": ["$data.full_name", "N/A"]},
                "relevant_roles": "$data.relevant_roles",
                "experience_category": {"$ifNull": ["$data.experience_category", "Fresher"]},
                "experience_value": {
                    "$let": {
                        "vars": {
                            "exp": {
                                "$arrayElemAt": [
                                    {"$split": ["$data.experience_value", " "]},
                                    0
                                ]
                            }
                        },
                        "in": {
                            "$cond": {
                                "if": {"$eq": ["$$exp", None]},
                                "then": "0",
                                "else": "$$exp"
                            }
                        }
                    }
                },
                "LocationCategory": {"$ifNull": ["$data.LocationCategory", "N/A"]},
                "LocationExact": {"$ifNull": ["$data.LocationExact", "N/A"]},
                "QualificationCategory": {"$ifNull": ["$data.QualificationCategory", "N/A"]},
                "QualificationExact": {"$ifNull": ["$data.QualificationExact", "N/A"]},
                "certifications": {
                    "$cond": {
                        "if": {"$eq": [{"$size": {"$ifNull": ["$data.certifications", []]}}, 0]},
                        "then": "No",
                        "else": "$data.certifications"
                    }
                },
                "file_name": 1,
                "reference_name": 1
            }
        })

        candidates = list(resumes_collection.aggregate(pipeline))
        # Manually encode the data to handle ObjectId and other non-JSON-serializable types
        def custom_encoder(obj):
            if isinstance(obj, ObjectId):
                return str(obj)
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        candidates_json = json.dumps(candidates, default=custom_encoder)

        return {"candidates": json.loads(candidates_json)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching candidates: {str(e)}")

@candidate_router.get("/locations")
async def get_locations(current_user: dict = Depends(get_current_user)):
    try:
        pipeline = [
            {"$group": {"_id": "$data.LocationCategory", "count": {"$sum": 1}}},
            {"$match": {"_id": {"$ne": None}}},
            {"$sort": {"_id": 1}}
        ]
        locations = [doc["_id"] for doc in resumes_collection.aggregate(pipeline) if doc["_id"] != "N/A"]
        return {"locations": locations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching locations: {str(e)}")