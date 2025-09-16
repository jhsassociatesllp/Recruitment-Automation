from fastapi import APIRouter, HTTPException, Request
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from fastapi.responses import PlainTextResponse, StreamingResponse
import os
from models.model import *
from database.db import *
import logging
from typing import List, Dict
from sse_starlette.sse import EventSourceResponse
import asyncio
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

whatsapp_router = APIRouter()

# Twilio configuration
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_client = Client(account_sid, auth_token)
twilio_number = os.getenv("TWILIO_PHONE_NUMBER")

# In-memory storage (replace with database in production)
messaged_candidates = {}


# Sending message logic through Twilio Messaging API
def send_whattsapp_message(to_number: str, body_text: str):
    try:
        message = twilio_client.messages.create(
            from_=f"whatsapp:{twilio_number}",
            body=body_text,
            to=f"whatsapp:{to_number}"
        )
        logger.info(f"Message sent to {to_number}: {message.body}")
        return message.sid
    except Exception as e:
        logger.error(f"Error sending message to {to_number}: {e}")
        raise

@whatsapp_router.post("/send-whatsapp-message/")
async def send_whatsapp_message(request: MessageRequest):
    successful_sids = []
    failed_candidates = []

    days = list(request.slots.keys())

    for candidate in request.candidates:
        candidate_id = candidate.candidate_id
        mobile = f"+91{candidate.mobile}"
        message = f"Hey {candidate.candidate_name}, this is JHS recruitment bot. You have been selected for the role {candidate.role}. Are you ready for an interview?"

        try:
            message_sid = send_whattsapp_message(mobile, message)

            messaged_collection.insert_one({
                "candidate_id": candidate_id,
                "name": candidate.candidate_name,
                "mobile": candidate.mobile,
                "email": candidate.email,
                "role": candidate.role,
                "hr_name": candidate.hr_name,
                "mode": request.mode,
                "slots": request.slots,
                "days": days,
                "status": "Messaged",
                "message_sid": message_sid
            })
            successful_sids.append(message_sid)
            logger.info(f"Candidate {candidate_id} stored in Messaged collection")
        except Exception as e:
            failed_candidates.append({"candidate_id": candidate_id, "error": str(e)})

    if successful_sids:
        return {"status": "messages_sent", "message_sids": successful_sids, "failed": failed_candidates}
    else:
        raise HTTPException(status_code=500, detail={"error": "All messages failed", "details": failed_candidates})

@whatsapp_router.post("/whatsapp-webhook/")
async def whatsapp_webhook(request: Request):
    form = await request.form()
    incoming_msg = form.get("Body", "").lower().strip()
    from_number = form.get("From", "")
    resp = MessagingResponse()

    candidate_data = messaged_collection.find_one({"mobile": from_number.replace("whatsapp:+91", "")})
    
    if not candidate_data:
        resp.message("Sorry, I couldn't identify you. Please contact HR.")
        return PlainTextResponse(str(resp), media_type="application/xml")

    candidate_id = candidate_data["candidate_id"]
    status = candidate_data.get("status")

    if status == "Messaged":
        if incoming_msg == "yes":
            messaged_collection.update_one({"candidate_id": candidate_id}, {"$set": {"status": "Replied"}})
            logger.info(f"Candidate {candidate_id} status updated to Replied")

            days = candidate_data["days"]
            if not days:
                resp.message("No interview days available. Please contact HR.")
                return PlainTextResponse(str(resp), media_type="application/xml")

            day_message = "Available days: " + ", ".join(day.capitalize() for day in days) + "\n"
            for i, day in enumerate(days, 1):
                day_message += f"If {day.capitalize()} press {i}\n"
            resp.message(day_message)
            await send_sse_update("messaged", {"candidate_id": candidate_id, "status": "Replied"})
            return PlainTextResponse(str(resp), media_type="application/xml")
        
        elif incoming_msg == "no":
            messaged_collection.update_one({"candidate_id": candidate_id}, {"$set": {"status": "Declined"}})
            logger.info(f"Candidate {candidate_id} status updated to Declined")
            resp.message("Thank you for your response. Please contact HR if you change your mind.")
            await send_sse_update("messaged", {"candidate_id": candidate_id, "status": "Declined"})
            return PlainTextResponse(str(resp), media_type="application/xml")
        
        else:
            resp.message("Invalid response. Please reply 'yes' or 'no' to proceed.")
            return PlainTextResponse(str(resp), media_type="application/xml")

    elif status == "Replied":
        try:
            day_index = int(incoming_msg) - 1
            days = candidate_data["days"]
            if day_index < 0 or day_index >= len(days):
                raise ValueError("Invalid day selection")
            
            selected_day = days[day_index]
            time_slots = candidate_data["slots"].get(selected_day, [])
            if not time_slots:
                resp.message(f"No time slots available for {selected_day.capitalize()}. Please contact HR.")
                return PlainTextResponse(str(resp), media_type="application/xml")

            messaged_collection.update_one(
                {"candidate_id": candidate_id},
                {"$set": {"status": "DaySelected", "selected_day": selected_day, "time_slots": time_slots}}
            )
            logger.info(f"Candidate {candidate_id} selected day: {selected_day}")

            slot_message = f"{selected_day.capitalize()} slots: " + ", ".join(time_slots) + "\n"
            for i, slot in enumerate(time_slots, 1):
                slot_message += f"For {slot} press {i}\n"
            resp.message(slot_message)
            await send_sse_update("messaged", {"candidate_id": candidate_id, "status": "DaySelected"})
            return PlainTextResponse(str(resp), media_type="application/xml")
        
        except ValueError:
            resp.message("Invalid response. Please press a number corresponding to the day.")
            return PlainTextResponse(str(resp), media_type="application/xml")

    elif status == "DaySelected":
        try:
            slot_index = int(incoming_msg) - 1
            selected_day = candidate_data["selected_day"]
            time_slots = candidate_data["time_slots"]
            if slot_index < 0 or slot_index >= len(time_slots):
                raise ValueError("Invalid time slot selection")
            
            selected_slot = time_slots[slot_index]

            scheduled_collection.insert_one({
                "candidate_id": candidate_id,
                "name": candidate_data["name"],
                "mobile": candidate_data["mobile"],
                "email": candidate_data["email"],
                "role": candidate_data["role"],
                "hr_name": candidate_data["hr_name"],
                "mode": candidate_data["mode"],
                "selected_day": selected_day,
                "selected_slot": selected_slot,
                "interview_status": "Scheduled",
                "rescheduled": False
            })
            logger.info(f"Candidate {candidate_id} interview scheduled: {selected_day} {selected_slot}")

            messaged_collection.delete_one({"candidate_id": candidate_id})
            logger.info(f"Candidate {candidate_id} removed from Messaged collection")

            resp.message(f"Your interview has been scheduled on {selected_day.capitalize()} at {selected_slot}. You can come.")
            await send_sse_update("scheduled", {"candidate_id": candidate_id, "status": "Scheduled", "day": selected_day, "slot": selected_slot})
            return PlainTextResponse(str(resp), media_type="application/xml")
        
        except ValueError:
            resp.message("Invalid response. Please press a number corresponding to the time slot.")
            return PlainTextResponse(str(resp), media_type="application/xml")

    resp.message("Invalid response. Please follow the instructions provided.")
    return PlainTextResponse(str(resp), media_type="application/xml")

async def send_sse_update(event_type: str, data: dict):
    # This function will be called by the event source to send updates
    async def event_generator():
        yield f"data: {{\"type\": \"{event_type}\", \"data\": {data}}}\n\n"
    return EventSourceResponse(event_generator())

@whatsapp_router.get("/events")
async def sse_events(request: Request):
    async def event_stream():
        while True:
            if await request.is_disconnected():
                break
            yield {"event": "keep-alive", "data": "ping"}
            await asyncio.sleep(30)  # Keep-alive ping every 30 seconds
    return EventSourceResponse(event_stream())

@whatsapp_router.get("/messaged/")
async def get_messaged():
    try:
        messaged = list(messaged_collection.find({}, {"_id": 0}))
        return {"messaged": messaged}
    except Exception as e:
        logger.error(f"Error fetching messaged candidates: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch messaged candidates")

@whatsapp_router.get("/scheduled/")
async def get_scheduled():
    try:
        scheduled = list(scheduled_collection.find({}, {"_id": 0}))
        return {"scheduled": scheduled}
    except Exception as e:
        logger.error(f"Error fetching scheduled candidates: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch scheduled candidates")

@whatsapp_router.post("/reschedule/")
async def reschedule_interview(request: RescheduleRequest):
    candidate_id = request.candidate_id
    candidate_data = scheduled_collection.find_one({"candidate_id": candidate_id})
    
    if not candidate_data:
        raise HTTPException(status_code=404, detail="Candidate not found in Scheduled collection")

    mobile = f"+91{candidate_data['mobile']}"
    days = list(request.slots.keys())
    selected_day = days[0]  # Use first day for simplicity
    selected_slot = request.slots[selected_day][0]  # Use first slot

    try:
        message = f"Your interview has been rescheduled to {selected_day.capitalize()} at {selected_slot}. You can come."
        message_sid = send_whattsapp_message(mobile, message)

        scheduled_collection.update_one(
            {"candidate_id": candidate_id},
            {
                "$set": {
                    "mode": request.mode,
                    "selected_day": selected_day,
                    "selected_slot": selected_slot,
                    "rescheduled": True
                }
            }
        )
        logger.info(f"Candidate {candidate_id} rescheduled to {selected_day} {selected_slot}")
        await send_sse_update("scheduled", {"candidate_id": candidate_id, "status": "Rescheduled", "day": selected_day, "slot": selected_slot})
        return {"status": "rescheduled", "message_sid": message_sid}
    except Exception as e:
        logger.error(f"Error rescheduling for {candidate_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to reschedule interview")