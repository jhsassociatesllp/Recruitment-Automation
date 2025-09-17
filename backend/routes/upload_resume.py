# from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
# from services.route_services import *  # Assuming extract_text_from_docx is here
# from database.db import *  # Assuming resumes_collection is here
# import uuid
# from datetime import datetime
# from database.auth import get_current_user
# from typing import List
# import openai
# from pinecone import Pinecone, ServerlessSpec
# import os
# from dotenv import load_dotenv
# import time
# import traceback
# from PyPDF2 import PdfReader
# from io import BytesIO
# import fitz  # PyMuPDF
# from PIL import Image
# import base64

# load_dotenv()

# # Set API keys
# openai.api_key = os.getenv("OPENAI_API_KEY")
# pinecone_api_key = os.getenv("PINECONE_API_KEY")
# print("OpenAI API Key loaded:", "valid" if openai.api_key else "not found")
# print("Pinecone API Key loaded:", "valid" if pinecone_api_key else "not found")

# # Initialize OpenAI client
# openai_client = openai.OpenAI(api_key=openai.api_key)

# # Initialize Pinecone client
# pc = Pinecone(api_key=pinecone_api_key)
# index_name = "resumes"
# if not pc.has_index(index_name):
#     pc.create_index(
#         name=index_name,
#         dimension=1536,  # Match text-embedding-3-small dimension
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )
#     time.sleep(5)  # Wait for index to be ready
# index = pc.Index(index_name)

# upload_router = APIRouter()

# def pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
#     """Convert PIL image to base64 data URL."""
#     buf = BytesIO()
#     img.save(buf, format=fmt)
#     b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
#     mime = f"image/{fmt.lower()}"
#     return f"data:{mime};base64,{b64}"

# def pdf_to_pil_images(file) -> List[Image.Image]:
#     """
#     Render each PDF page from UploadFile to a high-res PIL image.
#     """
#     file.seek(0)
#     pdf_file = fitz.open(stream=file.read(), filetype="pdf")
#     images = []
#     zoom = 2.0  # ~144 DPI*2 for clearer text
#     mat = fitz.Matrix(zoom, zoom)
#     for page_index in range(len(pdf_file)):
#         page = pdf_file[page_index]
#         pix = page.get_pixmap(matrix=mat, alpha=False)
#         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#         images.append(img)
#     return images

# def ocr_page_with_openai_dataurl(data_url: str, model: str = "gpt-4o-mini", max_tokens: int = 1500) -> str:
#     """
#     Send one page image (as data URL) to OpenAI Vision model and get plain text back.
#     """
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": (
#                         "You are an OCR engine. Extract ALL readable text from this resume page. "
#                         "Return ONLY the plain text in natural reading order (top-to-bottom, left-to-right). "
#                         "Do not add commentary, labels, bullet symbols, or summaries. Keep original line breaks where natural."
#                     ),
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": data_url}
#                 },
#             ],
#         }
#     ]

#     resp = openai_client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=0,
#         max_tokens=max_tokens,
#     )
#     return resp.choices[0].message.content.strip()

# def extract_text_from_scanned_pdf(file):
#     """
#     Extract text from scanned PDF using OpenAI Vision API.
#     """
#     content = ""
#     try:
#         pages = pdf_to_pil_images(file)
#         for i, img in enumerate(pages, start=1):
#             data_url = pil_to_data_url(img, fmt="PNG")
#             page_text = ocr_page_with_openai_dataurl(data_url, model="gpt-4o-mini", max_tokens=1500)
#             if page_text and page_text.strip():
#                 content += f"=== Page {i} ===\n{page_text}\n\n"
#                 print(f"Extracted text (Vision API) for page {i}: {page_text[:50]}...")
#             time.sleep(0.6)  # Light pacing for rate limits
#     except Exception as e:
#         print(f"Vision API extraction failed: {e}")
#     return content.strip() if content else ""

# def extract_text_from_pdf(file):
#     content = ""
#     try:
#         # Standard text extraction with PyPDF2
#         file.seek(0)
#         reader = PdfReader(file)
#         for page in reader.pages:
#             text = page.extract_text()
#             if text and text.strip():
#                 content += text + "\n"
#         if content.strip():
#             print(f"Extracted text (standard) for {file.name or 'unknown'}: {content[:50]}...")
#             return content
#     except Exception as e:
#         print(f"Standard extraction failed for {file.name or 'unknown'}: {e}")

#     # Fallback to Vision API for scanned PDFs
#     print(f"Falling back to Vision API for {file.name or 'unknown'} due to extraction failure.")
#     return extract_text_from_scanned_pdf(file)

# def normalize_location(location):
#     """Normalize location to standard city names."""
#     if not location:
#         return None
#     location = location.strip().lower()
#     if "mumbai" in location:
#         return "Mumbai" if "navi" not in location else "Navi Mumbai"
#     elif "pune" in location:
#         return "Pune"
#     return location.title()  # Capitalize other city names

# def map_experience(experience):
#     """Map experience to predefined categories."""
#     if not experience or "fresher" in experience.lower():
#         return "Fresher"
#     experience = experience.lower().replace("years", "").strip()
#     try:
#         years = float(experience.split()[0]) if experience.split() else 0
#         if years <= 1:
#             return "0–1 year"
#         elif years <= 3:
#             return "1–3 years"
#         elif years <= 5:
#             return "3–5 years"
#         else:
#             return "5 years"
#     except (ValueError, IndexError):
#         return "Fresher"  # Default if parsing fails

# def map_qualification(qualification):
#     """Map qualification to predefined categories."""
#     if not qualification:
#         return None
#     qualification = qualification.lower().strip()
#     if "bachelor" in qualification or "b.tech" in qualification or "b.e" in qualification:
#         return "Bachelor’s"
#     elif "master" in qualification or "m.tech" in qualification or "m.e" in qualification:
#         return "Master’s"
#     elif "mba" in qualification:
#         return "MBA"
#     elif "phd" in qualification:
#         return "PhD"
#     return None

# @upload_router.post("/upload-resumes/")
# async def upload_resumes(resumes: List[UploadFile] = File(...), reference_name: str = Form(...), current_user: dict = Depends(get_current_user)):
#     print(f"Reference Name: {reference_name}")
#     print(f"Resumes: {resumes}")
#     hr_id = str(current_user["_id"])
#     if not reference_name:
#         raise HTTPException(status_code=400, detail="Reference Name is mandatory.")

#     print(f"Current User: {current_user}")
#     allowed_types = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
#     max_size = 5 * 1024 * 1024  # 5MB in bytes

#     uploaded_resumes = []
#     for i, resume in enumerate(resumes):
#         print(f"Processing file {i+1}/{len(resumes)}: {resume.filename}")
#         if resume.content_type not in allowed_types:
#             print(f"Skipping invalid file type: {resume.filename}")
#             continue

#         file_content = await resume.read()
#         file_size = len(file_content)
#         if file_size > max_size:
#             print(f"Skipping large file: {resume.filename} (size: {file_size} bytes)")
#             continue

#         content = ""
#         resume.file.seek(0)
#         try:
#             if resume.content_type == "application/pdf":
#                 content = extract_text_from_pdf(resume.file)
#             else:
#                 content = extract_text_from_docx(resume.file)
#         except Exception as e:
#             print(f"Error extracting text from {resume.filename}: {e}")
#             content = ""  # Set to empty string if extraction fails

#         # Extract structured data using GPT-4o-mini
#         try:
#             prompt = """
#             Extract the following fields from the provided resume text in a structured JSON format:
#             - full_name: The candidate's full name
#             - relevant_roles: List of job roles the candidate is suitable for (e.g., ["Software Engineer", "Data Analyst"])
#             - experience: Years of experience (e.g., "2 years", "Fresher")
#             - highest_qualification: Highest educational qualification (e.g., "B.Tech", "MBA")
#             - age: Age of the candidate (if available, else null)
#             - skills: List of skills mentioned (e.g., ["Python", "JavaScript"])
#             - location: Candidate's location (if available, else null)
#             - certifications: List of certifications (if available, else [])

#             Return the output in JSON format. If a field cannot be extracted, use null or an empty list as appropriate.
#             Resume text:
#             {resume_text}
#             """
#             response = openai_client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant skilled in extracting structured data from text."},
#                     {"role": "user", "content": prompt.format(resume_text=content or "No content extracted")}
#                 ],
#                 response_format={"type": "json_object"}
#             )
#             extracted_data = json.loads(response.choices[0].message.content)
#             print(f"Extracted data for {resume.filename}: {extracted_data}")

#             # Normalize and map extracted data
#             extracted_data["location"] = normalize_location(extracted_data.get("location"))
#             extracted_data["experience"] = map_experience(extracted_data.get("experience"))
#             extracted_data["highest_qualification"] = map_qualification(extracted_data.get("highest_qualification"))
#         except Exception as e:
#             print(f"Error extracting data with GPT-4o-mini for {resume.filename}: {e}")
#             extracted_data = {
#                 "full_name": None,
#                 "relevant_roles": [],
#                 "experience": None,
#                 "highest_qualification": None,
#                 "age": None,
#                 "skills": [],
#                 "location": None,
#                 "certifications": []
#             }

#         resume_id = str(uuid.uuid4())
#         candidate_id = str(uuid.uuid4())

#         resume_doc = {
#             "_id": resume_id,
#             "candidate_id": candidate_id,
#             "hr_id": hr_id,
#             "reference_name": reference_name,
#             "content": content,
#             "file_data": file_content,  # Store original binary data
#             "file_type": "pdf" if resume.content_type == "application/pdf" else "docx",
#             "file_name": resume.filename,
#             "upload_date": datetime.utcnow().isoformat(),
#             "data": extracted_data,
#             "metadata": {
#                 "source": "website_upload",
#                 "size": file_size,
#                 "parsed_successfully": bool(content)
#             }
#         }
#         try:
#             resumes_collection.insert_one(resume_doc)
#             print(f"Stored resume: {resume.filename} with ID: {resume_id}")
#         except Exception as e:
#             print(f"Error storing resume {resume.filename} in MongoDB: {e}")
#             continue

#         try:
#             embedding_response = openai_client.embeddings.create(
#                 input=content or "",
#                 model="text-embedding-3-small"
#             )
#             embedding = embedding_response.data[0].embedding
#             print(f"Created embedding for {resume.filename}")
#         except Exception as e:
#             print(f"Error creating embedding for {resume.filename}: {e}")
#             continue

#         max_retries = 3
#         for attempt in range(max_retries):
#             try:
#                 print(f"Attempting to add embedding to Pinecone (Attempt {attempt + 1}/{max_retries})")
#                 start_time = time.time()
#                 index.upsert(
#                     vectors=[
#                         {
#                             "id": resume_id,
#                             "values": embedding,
#                             "metadata": {
#                                 "resume_id": resume_id,
#                                 "candidate_id": candidate_id,
#                                 "hr_id": hr_id,
#                                 "reference_name": reference_name,
#                                 "file_name": resume.filename
#                             }
#                         }
#                     ]
#                 )
#                 elapsed_time = time.time() - start_time
#                 print(f"Added embedding for resume: {resume.filename} to Pinecone in {elapsed_time:.2f} seconds")
#                 break
#             except Exception as e:
#                 print(f"Error adding embedding for {resume.filename} to Pinecone (Attempt {attempt + 1}): {str(e)}")
#                 print(f"Stack trace: {traceback.format_exc()}")
#                 if attempt < max_retries - 1:
#                     time.sleep(2 ** attempt)  # Exponential backoff
#                 else:
#                     print(f"Max retries reached for {resume.filename}, skipping")
#                     break

#         uploaded_resumes.append({
#             "resume_id": resume_id,
#             "candidate_id": candidate_id,
#             "file_name": resume.filename
#         })

#     if uploaded_resumes:
#         return {
#             "message": "Resumes uploaded successfully",
#             "status": 200,
#             "success": True,
#             "data": uploaded_resumes
#         }
#     else:
#         raise HTTPException(status_code=400, detail="No valid resumes uploaded.")




from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
# from services.route_services import *  # Assuming extract_text_from_docx is here
from database.db import *  # Assuming resumes_collection is here
import uuid
from datetime import datetime
from database.auth import get_current_user
from typing import List
import openai
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import time
import traceback
from PyPDF2 import PdfReader
from io import BytesIO
import fitz  # PyMuPDF
from PIL import Image
import base64
import docx

load_dotenv()

# Set API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
print("OpenAI API Key loaded:", "valid" if openai.api_key else "not found")
print("Pinecone API Key loaded:", "valid" if pinecone_api_key else "not found")

# Initialize OpenAI client
openai_client = openai.OpenAI(api_key=openai.api_key)

# Initialize Pinecone client
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

upload_router = APIRouter()

def pil_to_data_url(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL image to base64 data URL."""
    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"

def pdf_to_pil_images(file) -> List[Image.Image]:
    """
    Render each PDF page from UploadFile to a high-res PIL image.
    """
    file.seek(0)
    pdf_file = fitz.open(stream=file.read(), filetype="pdf")
    images = []
    zoom = 2.0  # ~144 DPI*2 for clearer text
    mat = fitz.Matrix(zoom, zoom)
    for page_index in range(len(pdf_file)):
        page = pdf_file[page_index]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

def ocr_page_with_openai_dataurl(data_url: str, model: str = "gpt-4o-mini", max_tokens: int = 1500) -> str:
    """
    Send one page image (as data URL) to OpenAI Vision model and get plain text back.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are an OCR engine. Extract ALL readable text from this resume page. "
                        "Return ONLY the plain text in natural reading order (top-to-bottom, left-to-right). "
                        "Do not add commentary, labels, bullet symbols, or summaries. Keep original line breaks where natural."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url}
                },
            ],
        }
    ]

    resp = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

def extract_text_from_scanned_pdf(file):
    """
    Extract text from scanned PDF using OpenAI Vision API.
    """
    content = ""
    try:
        pages = pdf_to_pil_images(file)
        for i, img in enumerate(pages, start=1):
            data_url = pil_to_data_url(img, fmt="PNG")
            page_text = ocr_page_with_openai_dataurl(data_url, model="gpt-4o-mini", max_tokens=1500)
            if page_text and page_text.strip():
                content += f"=== Page {i} ===\n{page_text}\n\n"
                print(f"Extracted text (Vision API) for page {i}: {page_text[:50]}...")
            time.sleep(0.6)  # Light pacing for rate limits
    except Exception as e:
        print(f"Vision API extraction failed: {e}")
    return content.strip() if content else ""

def extract_text_from_pdf(file):
    content = ""
    try:
        # Standard text extraction with PyPDF2
        file.seek(0)
        reader = PdfReader(file)
        for page in reader.pages:
            text = page.extract_text()
            if text and text.strip():
                content += text + "\n"
        if content.strip():
            print(f"Extracted text (standard) for {file.name or 'unknown'}: {content[:50]}...")
            return content
    except Exception as e:
        print(f"Standard extraction failed for {file.name or 'unknown'}: {e}")

    # Fallback to Vision API for scanned PDFs
    print(f"Falling back to Vision API for {file.name or 'unknown'} due to extraction failure.")
    return extract_text_from_scanned_pdf(file)

def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        content = "\n".join([para.text for para in doc.paragraphs if para.text])
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from DOCX: {str(e)}")

def convert_experience_to_years(experience):
    """Convert experience string to decimal years."""
    if not experience or "fresher" in experience.lower() or "no experience" in experience.lower():
        return 0.0
    try:
        parts = experience.lower().replace("years", "").replace("year", "").strip().split()
        years = float(parts[0])
        if len(parts) > 1 and "month" in parts[1]:
            months = float(parts[2]) if len(parts) > 2 else 0.0
            years += months / 12
        return round(years, 1)
    except (ValueError, IndexError):
        return 0.0

def map_experience_category(years):
    """Map decimal years to experience category."""
    if years == 0.0:
        return "Fresher"
    elif 0 < years <= 1:
        return "0–1 year"
    elif 1 < years <= 3:
        return "1–3 years"
    elif 3 < years <= 5:
        return "3–5 years"
    else:
        return ">5 years"

def normalize_location(location):
    """Normalize location to standard city names."""
    if not location:
        return {"LocationCategory": None, "LocationExact": None}
    location = location.strip().lower()
    city = location.split(",")[0].strip()  # Extract city from location string
    if "mumbai" in city:
        return {"LocationCategory": "Mumbai", "LocationExact": city}
    elif "navi mumbai" in city:
        return {"LocationCategory": "Navi Mumbai", "LocationExact": city}
    elif "pune" in city:
        return {"LocationCategory": "Pune", "LocationExact": city}
    else:
        return {"LocationCategory": city.title(), "LocationExact": city.title()}

def map_qualification(qualification):
    """Map qualification to predefined categories."""
    if not qualification:
        return {"QualificationCategory": None, "QualificationExact": None}
    qualification = qualification.lower().strip()
    if "bachelor" in qualification or "b.tech" in qualification or "b.e" in qualification:
        return {"QualificationCategory": "Bachelor's", "QualificationExact": qualification.title()}
    elif "master" in qualification or "m.tech" in qualification or "m.e" in qualification:
        return {"QualificationCategory": "Master's", "QualificationExact": qualification.title()}
    elif "mba" in qualification:
        return {"QualificationCategory": "MBA", "QualificationExact": qualification.title()}
    elif "phd" in qualification:
        return {"QualificationCategory": "PhD", "QualificationExact": qualification.title()}
    return {"QualificationCategory": None, "QualificationExact": qualification.title()}

@upload_router.post("/upload-resumes/")
async def upload_resumes(resumes: List[UploadFile] = File(...), reference_name: str = Form(...), current_user: dict = Depends(get_current_user)):
    print(f"Reference Name: {reference_name}")
    print(f"Resumes: {resumes}")
    hr_id = str(current_user["_id"])
    if not reference_name:
        raise HTTPException(status_code=400, detail="Reference Name is mandatory.")

    print(f"Current User: {current_user}")
    allowed_types = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    max_size = 5 * 1024 * 1024  # 5MB in bytes

    uploaded_resumes = []
    for i, resume in enumerate(resumes):
        print(f"Processing file {i+1}/{len(resumes)}: {resume.filename}")
        if resume.content_type not in allowed_types:
            print(f"Skipping invalid file type: {resume.filename}")
            continue

        file_content = await resume.read()
        file_size = len(file_content)
        if file_size > max_size:
            print(f"Skipping large file: {resume.filename} (size: {file_size} bytes)")
            continue

        content = ""
        resume.file.seek(0)
        try:
            if resume.content_type == "application/pdf":
                content = extract_text_from_pdf(resume.file)
            else:
                content = extract_text_from_docx(resume.file)
        except Exception as e:
            print(f"Error extracting text from {resume.filename}: {e}")
            content = ""  # Set to empty string if extraction fails

        # Extract structured data using GPT-4o-mini
        try:
            prompt = """
            Extract the following fields from the provided resume text in a structured JSON format:
            - full_name: The candidate's full name
            - relevant_roles: List of job roles the candidate is suitable for (e.g., ["Software Engineer", "Data Analyst"])
            - experience: Years of experience (e.g., "2 years 4 months", "Fresher")
            - highest_qualification: Highest educational qualification (e.g., "Bachelor of Commerce", "MBA")
            - age: Age of the candidate (if available, else null)
            - skills: List of skills mentioned (e.g., ["Python", "JavaScript"])
            - location: Candidate's location (e.g., "Thane, Maharashtra", "Pune")
            - certifications: List of certifications (if available, else [])

            Return the output in JSON format. If a field cannot be extracted, use null or an empty list as appropriate.
            Resume text:
            {resume_text}
            """
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant skilled in extracting structured data from text."},
                    {"role": "user", "content": prompt.format(resume_text=content or "No content extracted")}
                ],
                response_format={"type": "json_object"}
            )
            extracted_data = json.loads(response.choices[0].message.content)
            print(f"Extracted data for {resume.filename}: {extracted_data}")

            # Process and map extracted data
            years = convert_experience_to_years(extracted_data.get("experience"))
            experience_category = map_experience_category(years)
            location_data = normalize_location(extracted_data.get("location"))
            qualification_data = map_qualification(extracted_data.get("highest_qualification"))

            # Update extracted_data with structured fields
            extracted_data.update({
                "experience_category": experience_category,
                "experience_value": f"{years} years" if years > 0 else "0 years",
                **location_data,
                **qualification_data
            })
        except Exception as e:
            print(f"Error extracting data with GPT-4o-mini for {resume.filename}: {e}")
            extracted_data = {
                "full_name": None,
                "relevant_roles": [],
                "experience": None,
                "experience_category": "Fresher",
                "experience_value": "0 years",
                "highest_qualification": None,
                "qualification_category": None,
                "qualification_exact": None,
                "age": None,
                "skills": [],
                "location": None,
                "location_category": None,
                "location_exact": None,
                "certifications": []
            }

        resume_id = str(uuid.uuid4())
        candidate_id = str(uuid.uuid4())

        resume_doc = {
            "_id": resume_id,
            "candidate_id": candidate_id,
            "hr_id": hr_id,
            "reference_name": reference_name,
            "content": content,
            "file_data": file_content,  # Store original binary data
            "file_type": "pdf" if resume.content_type == "application/pdf" else "docx",
            "file_name": resume.filename,
            "upload_date": datetime.utcnow().isoformat(),
            "data": extracted_data,
            "metadata": {
                "source": "website_upload",
                "size": file_size,
                "parsed_successfully": bool(content)
            }
        }
        try:
            resumes_collection.insert_one(resume_doc)
            print(f"Stored resume: {resume.filename} with ID: {resume_id}")
        except Exception as e:
            print(f"Error storing resume {resume.filename} in MongoDB: {e}")
            continue

        try:
            embedding_response = openai_client.embeddings.create(
                input=content or "",
                model="text-embedding-3-small"
            )
            embedding = embedding_response.data[0].embedding
            print(f"Created embedding for {resume.filename}")
        except Exception as e:
            print(f"Error creating embedding for {resume.filename}: {e}")
            continue

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Attempting to add embedding to Pinecone (Attempt {attempt + 1}/{max_retries})")
                start_time = time.time()
                index.upsert(
                    vectors=[
                        {
                            "id": resume_id,
                            "values": embedding,
                            "metadata": {
                                "resume_id": resume_id,
                                "candidate_id": candidate_id,
                                "hr_id": hr_id,
                                "reference_name": reference_name,
                                "file_name": resume.filename
                            }
                        }
                    ]
                )
                elapsed_time = time.time() - start_time
                print(f"Added embedding for resume: {resume.filename} to Pinecone in {elapsed_time:.2f} seconds")
                break
            except Exception as e:
                print(f"Error adding embedding for {resume.filename} to Pinecone (Attempt {attempt + 1}): {str(e)}")
                print(f"Stack trace: {traceback.format_exc()}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"Max retries reached for {resume.filename}, skipping")
                    break

        uploaded_resumes.append({
            "resume_id": resume_id,
            "candidate_id": candidate_id,
            "file_name": resume.filename
        })

    if uploaded_resumes:
        return {
            "message": "Resumes uploaded successfully",
            "status": 200,
            "success": True,
            "data": uploaded_resumes
        }
    else:
        raise HTTPException(status_code=400, detail="No valid resumes uploaded.")