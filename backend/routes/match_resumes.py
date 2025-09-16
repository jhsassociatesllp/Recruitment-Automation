from httpx import post
from fastapi import APIRouter, HTTPException
from database.db import *
from services.route_services import *
from models.model import *
import uuid
from bson import ObjectId
from copy import deepcopy
from fastapi.encoders import jsonable_encoder
import openai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from database.auth import *
import time
from mimetypes import guess_type
from fastapi.responses import FileResponse

load_dotenv()

match_router = APIRouter()

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)

# Pinecone setup
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
index_name = "resumes"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,  # Match text-embedding-3-small dimension
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    time.sleep(5)  # Wait for index to be ready
index = pc.Index(index_name)

# Endpoint to process JD and match resumes
@match_router.post("/match-resumes/")
async def match_resumes(request: MatchRequest, current_user: dict = Depends(get_current_user)):
    print(f"Current user: {current_user}")  # Debug current user
    print(f"Request data: {request}")  # Debug request
    threshold = request.threshold if request.threshold is not None else 60.0
    matched_candidates = []

    # Embed the JD
    jd_embedding_response = openai.embeddings.create(
        input=request.jd,
        model="text-embedding-3-small"
    )
    jd_embedding = jd_embedding_response.data[0].embedding

    # Query Pinecone for top 20 matches
    results = index.query(
        vector=jd_embedding,
        top_k=20,
        include_metadata=True,
        include_values=False
    )
    print(f"Pinecone query results: {results}")  # Debug Pinecone results

    # Extract top matches
    top_matches = []
    for match in results['matches']:
        resume_id = match['id']
        metadata = match['metadata']
        resume_doc = resumes_collection.find_one({"_id": resume_id})
        if resume_doc:
            top_matches.append({
                "resume_id": resume_id,
                "candidate_id": metadata['candidate_id'],
                "content": resume_doc['content'],
                "reference": metadata.get('reference_name', 'N/A'),
                "file_name": metadata['file_name'],
                "similarity_score": 1 - match['score']  # Convert distance to similarity
            })

    if not top_matches:
        print("No top matches found in Pinecone")  # Debug
        return {"matched_candidates": []}

    # Prepare prompt for GPT-4o-mini to score and rank
    prompt = f"""
    Job Description: {request.jd}
    Role: {request.role}

    Below are candidate resumes. Score each on a scale of 0-100 based on fit to the JD and role. Return a JSON array with the following fields for each candidate: 
    - candidate_id (use the provided candidate_id)
    - name
    - email
    - phone
    - role
    - score
    - reference (use the provided reference)
    - status (set to "New" for now, to be updated later)

    Resumes:
    {''.join([f"Resume {i+1} (candidate_id: {match['candidate_id']}, reference: {match['reference']}): {match['content']}\n\n" for i, match in enumerate(top_matches)])}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful HR assistant that scores resumes based on job descriptions. Return a valid JSON array with the specified fields, using the provided candidate_id and reference."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=1500
    )
    print(f"LLM response: {response.choices[0].message.content}")  # Debug LLM response

    # Parse the LLM response, handling code block markers
    raw_response = response.choices[0].message.content.strip()
    print(f"Raw response after strip: {raw_response}")  # Debug
    try:
        # Remove ```json and ``` if present
        if raw_response.startswith("```json") and raw_response.endswith("```"):
            raw_response = raw_response[len("```json"):].rstrip("```").strip()
        matched_candidates = json.loads(raw_response)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}")  # Debug
        raise HTTPException(status_code=500, detail="Error parsing LLM response: Invalid JSON format")
    except Exception as e:
        print(f"Unexpected parsing error: {str(e)}")  # Debug
        raise HTTPException(status_code=500, detail=f"Error parsing LLM response: {str(e)}")

    # Update status based on messaged and scheduled collections
    for candidate in matched_candidates:
        candidate_id = candidate.get("candidate_id")
        # Search in messaged collection
        messaged_doc = messaged_collection.find_one({"candidate_id": candidate_id})
        # Search in scheduled collection
        scheduled_doc = scheduled_collection.find_one({"candidate_id": candidate_id})

        if scheduled_doc and messaged_doc:
            # Use status from scheduled if in both
            candidate["status"] = scheduled_doc.get("status", "New")
        elif messaged_doc:
            # Use status from messaged if only there
            candidate["status"] = messaged_doc.get("status", "New")
        else:
            # Not in either, set to New
            candidate["status"] = "New"

        # Filter by threshold
        if candidate.get("score", 0) < threshold:
            continue

    # Ensure all candidates are JSON-serializable
    matched_candidates = jsonable_encoder(matched_candidates)

    # Debug: Print final matched_candidates list
    print(f"Returning matched_candidates: {json.dumps(matched_candidates, indent=2)}")

    return {"matched_candidates": matched_candidates}

@match_router.get("/get-resume/{candidate_id}")
async def get_resume(candidate_id: str, download: bool = False, current_user: dict = Depends(get_current_user)):
    resume_doc = resumes_collection.find_one({"candidate_id": candidate_id})
    if not resume_doc:
        raise HTTPException(status_code=404, detail="Resume not found")

    file_data = resume_doc.get("file_data")
    file_name = resume_doc["file_name"]
    if not file_data or not file_name:
        raise HTTPException(status_code=500, detail="Resume file data or filename not found")

    # Determine MIME type based on file extension
    content_type, _ = guess_type(file_name)
    if not content_type:
        if file_name.lower().endswith('.pdf'):
            content_type = "application/pdf"
        elif file_name.lower().endswith('.doc') or file_name.lower().endswith('.docx'):
            content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        else:
            content_type = "application/octet-stream"

    import tempfile
    with tempfile.NamedTemporaryFile(mode='wb', suffix=os.path.splitext(file_name)[1], delete=False) as temp_file:
        temp_file.write(file_data)
        temp_file_path = temp_file.name

    if download:
        return FileResponse(
            path=temp_file_path,
            filename=file_name,
            media_type=content_type,
            # background=True
        )
    else:
        return FileResponse(
            path=temp_file_path,
            filename=file_name,
            media_type=content_type,
            # background=True
        )

