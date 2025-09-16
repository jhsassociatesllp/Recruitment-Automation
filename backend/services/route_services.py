from fastapi import HTTPException
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pywhatkit
from email.mime.text import MIMEText
import os
import smtplib
import PyPDF2
import docx
import re
import json 
import torch
from openai import AzureOpenAI
from decouple import config
from bson import ObjectId

# Azure OpenAI setup
azure_client = AzureOpenAI(
    api_key="7219267fcc1345cabcd25ac868c686c1",
    api_version="2024-02-15-preview",
    azure_endpoint="https://stock-agent.openai.azure.com/"
)
deployment_name = "model-4o"

# LLaMA 3.1-8B setup
# model_path = "C:/Users/vasu.gadde/.llama/checkpoints/Llama3.1-8B"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#     device_map="auto"
# )
# llm = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     device=0 if torch.cuda.is_available() else -1,
#     max_new_tokens=200
# )


# Helper function to ensure JSON-serializable dictionary
def ensure_json_serializable(data):
    try:
        json.dumps(data)
        return data
    except TypeError as e:
        raise ValueError(f"Data is not JSON-serializable: {str(e)}")

# Helper function to convert MongoDB document to JSON-serializable format
def convert_mongo_document(doc):
    print("Converting MongoDB document to JSON-serializable format")
    if isinstance(doc, dict):
        return {k: str(v) if isinstance(v, ObjectId) else v for k, v in doc.items()}
    return doc

# Helper function to match resume with JD using Azure OpenAI
def match_resume_with_jd(resume_text: str, jd: str) -> float:
    prompt = f"""
    Compare the following resume with the job description and return a match score between 0 and 100 based on their compatibility. Provide the score as a single number in the format: [SCORE]
    
    Resume: {resume_text[:2000]}  # Truncate to avoid token limit
    Job Description: {jd}
    
    Output format: [SCORE]
    """
    try:
        response = azure_client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are an assistant that evaluates resumes against job descriptions and provides a numerical match score."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.3
        )
        output = response.choices[0].message.content.strip()
        
        # Extract score using regex
        match = re.search(r'\[(\d+\.?\d*)\]', output)
        if match:
            score = float(match.group(1))
            return min(max(score, 0), 100)
        else:
            raise ValueError(f"Could not extract valid score from Azure OpenAI output: {output}")
    except Exception as e:
        raise ValueError(f"Error in Azure OpenAI score generation: {str(e)}")

# Helper function to extract candidate details using Azure OpenAI
def extract_candidate_details(resume_text: str) -> dict:
    print("In extract candidate details function")
    prompt = f"""
    Extract the name, phone, and email from the following resume. Return the details in JSON format:
    
    Resume: {resume_text[:2000]}  # Truncate to avoid token limit
    
    Output format:
    ```json
    {{
        "name": "string",
        "phone": "string",
        "email": "string"
    }}
    ```
    """
    try:
        response = azure_client.chat.completions.create(
            model=deployment_name,
            messages=[
                {"role": "system", "content": "You are an assistant that extracts structured information from resumes."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        output = response.choices[0].message.content.strip()
        
        # Extract JSON from output
        json_match = re.search(r'```json\n([\s\S]*?)\n```', output)
        if json_match:
            details = json.loads(json_match.group(1))
            return {
                "name": details.get("name", ""),
                "phone": details.get("phone", ""),
                "email": details.get("email", "")
            }
        else:
            raise ValueError("No valid JSON found in Azure OpenAI output")
    except Exception as e:
        raise ValueError(f"Failed to parse JSON from Azure OpenAI output: {output}, error: {str(e)}")
    
    
# # Helper function to match resume with JD using LLaMA 3.1-8B
# def match_resume_with_jd(resume_text: str, jd: str) -> float:
#     prompt = f"""
#     Compare the following resume with the job description and return a match score between 0 and 100 based on their compatibility. Provide the score as a single number in the format: [SCORE]
    
#     Resume: {resume_text[:2000]}  # Truncate to avoid token limit
#     Job Description: {jd}
    
#     Output format: [SCORE]
#     """
#     try:
#         result = llm(prompt, max_length=50, num_return_sequences=1)
#         output = result[0]['generated_text'].strip()
        
#         # Extract score using regex
#         match = re.search(r'\[(\d+\.?\d*)\]', output)
#         if match:
#             score = float(match.group(1))
#             return min(max(score, 0), 100)
#         else:
#             raise ValueError(f"Could not extract valid score from LLaMA output: {output}")
#     except Exception as e:
#         raise ValueError(f"Error in LLaMA score generation: {str(e)}")

# # Helper function to extract candidate details using LLaMA 3.1-8B
# def extract_candidate_details(resume_text: str) -> dict:
#     print("In extract candidate details function")
#     prompt = f"""
#     Extract the name, phone, and email from the following resume. Return the details in JSON format:
    
#     Resume: {resume_text[:2000]}  # Truncate to avoid token limit
    
#     Output format:
#     ```json
#     {{
#         "name": "string",
#         "phone": "string",
#         "email": "string"
#     }}
#     ```
#     """
#     try:
#         result = llm(prompt, max_length=200, num_return_sequences=1)
#         output = result[0]['generated_text'].strip()
        
#         # Extract JSON from output
#         json_match = re.search(r'```json\n([\s\S]*?)\n```', output)
#         if json_match:
#             details = json.loads(json_match.group(1))
#             return {
#                 "name": details.get("name", ""),
#                 "phone": details.get("phone", ""),
#                 "email": details.get("email", "")
#             }
#         else:
#             raise ValueError("No valid JSON found in LLaMA output")
#     except Exception as e:
#         raise ValueError(f"Failed to parse JSON from LLaMA output: {output}, error: {str(e)}")
    
    
# Helper function to send WhatsApp message
# def send_whatsapp_message(phone: str, message: str):
#     try:
#         pywhatkit.sendwhatmsg_instantly(phone, message)
#     except Exception as e:
#         print(f"Failed to send WhatsApp message: {e}")

# Helper function to send email
# def send_email(to_email: str, subject: str, body: str):
#     msg = MIMEText(body)
#     msg['Subject'] = subject
#     msg['From'] = os.getenv("EMAIL_FROM")
#     msg['To'] = to_email

#     with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
#         server.login(os.getenv("EMAIL_FROM"), os.getenv("EMAIL_PASSWORD"))
#         server.send_message(msg)

from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
import io

def extract_text_from_pdf(file):
    content = ""
    # Try standard text extraction first
    try:
        file.seek(0)  # Reset pointer
        reader = PdfReader(file)
        for page in reader.pages:
            content += page.extract_text() + "\n"
        if content.strip():
            return content
    except Exception as e:
        print(f"Standard extraction failed: {e}")

    # Fall back to OCR
    try:
        file.seek(0)  # Reset pointer
        images = convert_from_bytes(file.read())
        for image in images:
            content += pytesseract.image_to_string(image) + "\n"
    except Exception as e:
        print(f"OCR extraction failed: {e}")

    return content

# Helper function to extract text from PDF
# def extract_text_from_pdf(file):
#     try:
#         pdf_reader = PyPDF2.PdfReader(file)
#         content = ""
#         for page in pdf_reader.pages:
#             text = page.extract_text()
#             if text:
#                 content += text + "\n"
#         return content
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")

# Helper function to extract text from DOCX
def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        content = "\n".join([para.text for para in doc.paragraphs if para.text])
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from DOCX: {str(e)}")