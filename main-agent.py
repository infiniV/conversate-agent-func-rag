import os
import logging
import asyncio
from typing import AsyncIterable
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm, tokenize ,JobProcess
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero,cartesia
# from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage, Settings
from llama_index.core.schema import MetadataMode
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from supabase import create_client, Client
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from phonenumbers import parse as parse_phone, is_valid_number, format_number, PhoneNumberFormat
from functools import lru_cache

load_dotenv()
logger = logging.getLogger("conversate-agent")

# # Set HuggingFace embedding model
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5",)
# # LlamaIndex setup
# PERSIST_DIR = "./rag-storage"
# if not os.path.exists(PERSIST_DIR):
#     documents = SimpleDirectoryReader("data").load_data()
#     index = VectorStoreIndex.from_documents(documents)
#     index.storage_context.persist(persist_dir=PERSIST_DIR)
# else:
#     storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
#     index = load_index_from_storage(storage_context)

# Supabase setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


@dataclass
class BookingInfo:
    name: str
    phone: str
    time: str
    seat_number: int
    booking_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_initialized: bool = False  # Track if start_booking was called

class AssistantFunctions(llm.FunctionContext):
    """Functions for the booking assistant"""
    def __init__(self):
        super().__init__()
        self.booking_cache = {}
        self.current_booking_state = {}
        self.pronunciation_map = {
            "seat": "<<s|iː|t>>",
            "booking": "<<b|ʊ|k|ɪ|ŋ>>",
            "reference": "<<r|ɛ|f|ə|r|ə|n|s>>",
            "AM": "<<eɪ|ɛm>>",
            "PM": "<<piː|ɛm>>"
        }

    def _validate_phone(self, phone: str) -> tuple[bool, str]:
        try:
            clean_phone = phone.replace('-', '').replace(' ', '')
            if clean_phone.startswith('03'):
                clean_phone = '+92' + clean_phone[1:]
            elif clean_phone.startswith('3'):
                clean_phone = '+92' + clean_phone
            elif not clean_phone.startswith('+'):
                clean_phone = '+' + clean_phone
            parsed = parse_phone(clean_phone)
            if is_valid_number(parsed):
                return True, format_number(parsed, PhoneNumberFormat.E164)
        except Exception as e:
            logger.error(f"Phone validation error: {e}")
        return False, ""

    def _format_for_pronunciation(self, text: str) -> str:
        phone_pattern = r'\+?(?:92|1)?[0-9]{10,}'
        def phone_replace(match):
            num = match.group()
            if 'Pakistan' in num or '92' in num:
                parts = [num[i:i+3] for i in range(0, len(num), 3)]
                return ' - '.join(parts)
            return ' '.join([num[i:i+2] for i in range(0, len(num, 2))])
        text = re.sub(phone_pattern, phone_replace, text)
        time_pattern = r'\b(\d{1,2}):(\d{2})\b'
        def time_replace(match):
            hour, minute = match.groups()
            hour = int(hour)
            if hour == 0:
                return f"12 {minute} <<eɪ|ɛm>>"
            elif hour < 12:
                return f"{hour} {minute} <<eɪ|ɛm>>"
            elif hour == 12:
                return f"12 {minute} <<piː|ɛm>>"
            else:
                return f"{hour - 12} {minute} <<piː|ɛm>>"
        text = re.sub(time_pattern, time_replace, text)
        return tokenize.utils.replace_words(text=text, replacements=self.pronunciation_map)

    @lru_cache(maxsize=128)
    def _get_available_times_cached(self) -> list:
        try:
            response = supabase.table("lahore").select("time").execute()
            return response.data
        except Exception as e:
            logger.error(f"Error fetching times: {e}")
            return []

    def _get_closest_time(self, times: list) -> str:
        now = datetime.now()
        min_time = now + timedelta(hours=1)
        max_time = now + timedelta(hours=2)
        valid_times = []
        for time_str in times:
            time_obj = datetime.strptime(time_str, "%H:%M").replace(
                year=now.year, month=now.month, day=now.day
            )
            if min_time <= time_obj <= max_time:
                valid_times.append((time_obj, time_str))
        if not valid_times:
            return None
        return min(valid_times, key=lambda x: abs(x[0] - now))[1]

    @llm.ai_callable()
    async def get_next_available_slot(self) -> str:
        data = self._get_available_times_cached()
        if not data:
            return "Currently, there are no available time slots."
        times = sorted(set(slot["time"] for slot in data))
        closest_time = self._get_closest_time(times)
        if not closest_time:
            return "No available time slots in the next 2 hours. Would you like to check other times?"
        try:
            response = supabase.table("lahore").select("*").eq("time", closest_time).execute()
            data = response.data
            if not data:
                return f"No seats available for the next available time at {closest_time}."
            for i in range(45):
                if data[0][f"seat_{i+1}"] == 1:
                    return f"Next available slot is at {closest_time} with seat {i+1} available. Would you like to book this?"
            return f"Found time slot at {closest_time} but all seats are taken. Would you like to check other times?"
        except Exception as e:
            logger.error(f"Error checking availability: {e}")
            return "Sorry, I couldn't check the next available slot. Please try again."

    @llm.ai_callable()
    async def start_booking(self, name: str, phone: str) -> str:
        if len(name.strip()) < 2:
            return "The name you provided is too short. Please provide a valid name with at least 2 characters."
        is_valid, formatted_phone = self._validate_phone(phone)
        if not is_valid:
            return "The phone number you provided is invalid. Please provide a valid phone number in the format 03XX-XXXXXXX or +92XXX-XXXXXXX."
        self.current_booking_state = {
            "initialized": True,
            "phone": formatted_phone,
            "name": name.strip()
        }
        self.booking_cache[formatted_phone] = BookingInfo(
            name=name.strip(),
            phone=formatted_phone,
            time="",
            seat_number=0,
            is_initialized=True
        )
        return f"Thanks {name}! I've registered your details. Would you like me to check available time slots for you?"

    @llm.ai_callable()
    async def get_available_times(self, start_time: str = None, end_time: str = None) -> str:
        try:
            data = self._get_available_times_cached()
            if not data:
                return "No time slots available."
            times = sorted(set(slot["time"] for slot in data))
            current_hour = datetime.now().hour
            times = [t for t in times if int(t.split(':')[0]) >= current_hour]
            if not times:
                return "No slots available for today."
            next_times = times[:3]
            return f"Next available times are: {', '.join(next_times)}. Would you like to book any of these slots?"
        except Exception as e:
            logger.error(f"Error getting times: {e}")
            return "Sorry, I'm having trouble checking available times. Please try again."

    @llm.ai_callable()
    async def get_free_seats(self, time: str) -> str:
        try:
            response = supabase.table("lahore").select("*").eq("time", time).execute()
            data = response.data
            if not data:
                return f"No seats available for {time}."
            available_seats = [i+1 for i in range(45) if data[0][f"seat_{i+1}"] == 1]
            if not available_seats:
                return f"All seats taken for {time}."
            total_available = len(available_seats)
            first_seat = available_seats[0]
            if total_available == 1:
                return f"Seat {first_seat} available at {time}."
            else:
                return f"Seat {first_seat} and {total_available-1} more available at {time}."
        except Exception as e:
            logger.error(f"Error fetching seats: {e}")
            return "Sorry, couldn't check seats. Please try again."

    @llm.ai_callable()
    async def book_seat(self, phone: str, time: str, seat_number: int) -> str:
        is_valid, formatted_phone = self._validate_phone(phone)
        if not is_valid:
            return "Please provide a valid phone number before booking."
        if not self.current_booking_state.get("initialized"):
            return "Please provide your name and phone number first using the start booking process."
        booking_info = self.booking_cache.get(formatted_phone)
        if not booking_info or not booking_info.is_initialized:
            return "Please provide your name and phone number first using the start booking process."
        try:
            seat_column = f"seat_{seat_number}"
            response = supabase.table("lahore").select(seat_column).eq("time", time).execute()
            data = response.data
            if not data or data[0][seat_column] == 0:
                return f"Seat {seat_number} at {time} is not available. Please choose a different seat or time slot."
            supabase.table("lahore").update({seat_column: 0}).eq("time", time).execute()
            booking_info = self.booking_cache.get(formatted_phone, BookingInfo(
                name="",
                phone=formatted_phone,
                time=time,
                seat_number=seat_number
            ))
            booking_info.time = time
            booking_info.seat_number = seat_number
            supabase.table("bookings").insert({
                "name": booking_info.name,
                "phone": booking_info.phone,
                "time": booking_info.time,
                "seat": booking_info.seat_number,
                "booking_time": booking_info.booking_time
            }).execute()
            self.booking_cache[formatted_phone] = booking_info
            return (f"Your seat {seat_number} at {time} has been successfully booked. "
                    f"Your booking reference is {formatted_phone}.")
        except Exception as e:
            logger.error(f"Booking error: {e}")
            return "Sorry, there was an error with your booking. Please try again later."

    @llm.ai_callable()
    async def view_bookings(self, phone: str) -> str:
        is_valid, formatted_phone = self._validate_phone(phone)
        if not is_valid:
            return "The phone number you provided is invalid. Please provide a valid phone number."
        try:
            response = supabase.table("bookings").select("*").eq("phone", formatted_phone).execute()
            data = response.data
            if not data:
                return "No bookings found for the provided phone number."
            bookings = [f"Seat {b['seat']} at {b['time']}" for b in data]
            return f"Your bookings are: {', '.join(bookings)}"
        except Exception as e:
            logger.error(f"Error fetching bookings: {e}")
            return "Sorry, I couldn't retrieve your bookings. Please try again later."

    @llm.ai_callable()
    async def cancel_booking(self, phone: str, time: str, seat_number: int) -> str:
        is_valid, formatted_phone = self._validate_phone(phone)
        if not is_valid:
            return "The phone number you provided is invalid. Please provide a valid phone number."
        
        try:
            response = supabase.table("bookings").select("*").eq("phone", formatted_phone).eq("time", time).eq("seat", seat_number).execute()
            data = response.data
            if not data:
                return f"No booking found for phone number {phone} at {time} for seat {seat_number}."
            
            # Delete the booking
            supabase.table("bookings").delete().eq("phone", formatted_phone).eq("time", time).eq("seat", seat_number).execute()
            
            # Update the seat availability
            seat_column = f"seat_{seat_number}"
            supabase.table("lahore").update({seat_column: 1}).eq("time", time).execute()
            
            return f"Your booking for seat {seat_number} at {time} has been successfully canceled."
        except Exception as e:
            logger.error(f"Cancel booking error: {e}")
            return "Sorry, there was an error canceling your booking. Please try again later."

async def entrypoint(ctx: JobContext):
    fnc_ctx = AssistantFunctions()

    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a helpful booking assistant for a venue management system. "
            "Your interface with users will be voice. "
            "Keep responses short and clear. Avoid using unpronounceable punctuation. "
            "Always Use am and pm for time slots. "
            "Always collect name and phone number first using start_booking before any other actions. "
            "After start_booking succeeds, guide user to: "
            "1. Check available times (use get_available_times) "
            "2. Select a time slot "
            "3. View available seats (use get_free_seats) "
            "4. Book a seat (use book_seat) "
            "If user tries to skip steps, ask for required information first."
        )
    )

    def _before_tts_cb(agent: VoicePipelineAgent, text: str | AsyncIterable[str]) -> str | AsyncIterable[str]:
        if isinstance(text, str):
            return fnc_ctx._format_for_pronunciation(text)
        return text

    # async def _enrich_with_rag(agent: VoicePipelineAgent, chat_ctx: llm.ChatContext):
    #     user_msg = chat_ctx.messages[-1]
    #     query_engine = index.as_query_engine(use_async=True, llm=agent.llm)
    #     try:
    #         result = await query_engine.aquery(user_msg.content)
    #         if result:
    #             rag_msg = llm.ChatMessage.create(
    #                 text=f"Additional context:\n{str(result)}",
    #                 role="assistant"
    #             )
    #             chat_ctx.messages.insert(-1, rag_msg)
    #     except Exception as e:
    #         logger.error(f"RAG enrichment failed: {e}")

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()

    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(
            model="nova-2-phonecall"
        ),
        llm=openai.LLM().with_groq(
            model="llama3-groq-70b-8192-tool-use-preview",
        ),
        tts=cartesia.TTS(),
        chat_ctx=initial_ctx,
        fnc_ctx=fnc_ctx,
        before_tts_cb=_before_tts_cb,
        max_nested_fnc_calls=5,  # Keep this at 3 for proper function chaining
        preemptive_synthesis=True,
        interrupt_speech_duration=1.0,  # Increase interrupt threshold
        min_endpointing_delay=1.0  # Increase endpoint delay
    )
    
    # Add event handlers
    def on_function_calls_collected(data):
        logger.debug(f"Function calls collected: {data}")
        # Reset state if needed
        if data and any(call.function_info.name == "start_booking" for call in data):
            fnc_ctx.current_booking_state = {}
    
    def on_function_calls_finished(data):
        logger.debug(f"Function calls finished: {data}")
        
    agent.on("function_calls_collected", on_function_calls_collected)
    agent.on("function_calls_finished", on_function_calls_finished)

    # Add retry logic for connection issues
    async def start_agent_with_retry(max_retries=3):
        for attempt in range(max_retries):
            try:
                agent.start(ctx.room, participant)
                await agent.say(
                    "Hello! How may I help you today?",
                    allow_interruptions=False
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1)

    await start_agent_with_retry()

def prewarm_process(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load(activation_threshold=0.3)
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm_process,
        ),
    )