# main.py (Updated: Naam Siya & Google Sheets hat gaya)
import os
import logging
import requests
import asyncio
import uuid
import pytz
import traceback
import random
from collections import defaultdict
from datetime import datetime
import psutil
import json
import re
# --- GSPREAD IMPORTS HATA DIYE GAYE ---
import google.generativeai as genai
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup, ChatPermissions
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, 
    ContextTypes, CallbackQueryHandler
)
from dotenv import load_dotenv
import time
import lyricsgenius as lg
import urllib.parse

# --- NEW: Imports for Speech-to-Text ---
import speech_recognition as sr
from pydub import AudioSegment

# --- NEW: Imports for YouTube Download ---
import yt_dlp
from yt_dlp.utils import DownloadError, YoutubeDLError 
import io
import glob
# --- NEW: IMPORT THE GAME MODULE ---
# ⚠️ Note: game.py ab crash ho sakta hai kyunki Sheets hata diya gaya hai
# import game 

# Load environment variables from .env file
load_dotenv()

# --- Environment Variables ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5"),
]
GEMMA_API_KEY = os.getenv("GEMMA_API_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
try:
    BROADCAST_ADMIN_ID = int(os.getenv("BROADCAST_ADMIN_ID"))
except (ValueError, TypeError):
    BROADCAST_ADMIN_ID = 0
    logging.error("BROADCAST_ADMIN_ID is missing or not a valid number. Broadcast functionality will be disabled.")

STABLE_HORDE_API_KEY = os.getenv("STABLE_HORDE_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY") # Optional for pytube

# --- NEW: API Keys for New Commands ---
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY") # From openweathermap.org
VISUALCROSSING_API_KEY = os.getenv("VISUALCROSSING_API_KEY") # From visualcrossing.com
TINYURL_API_KEY = os.getenv("TINYURL_API_KEY") # From tinyurl.com

# --- NEW: API Key for DeepSeek via OpenRouter ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")# From openrouter.ai

INVIDIOUS_BASE_URL = os.getenv("INVIDIOUS_BASE_URL")

# --- Global Stats Variables ---
start_time = datetime.now()
total_messages_processed = 0
known_users = set() # Ab yeh sirf session ke liye hai
# <--- Naya Global Variable Couple command ke liye --->
chat_members = defaultdict(dict)
# --- NEW: Global variable for the guessing game ---
guessing_games = defaultdict(dict)


# --- Sudo User Management ---
# ⚠️ IMPORTANT: Google Sheets hat gaya hai. Apne Sudo User IDs yahan daalein!
sudo_users = {12345678, 98765432} # <-- ⚠️ APNA ADMIN ID YAHAN DAALEIN
# --- Google Sheets se related variables hata diye gaye ---


# --- Logging Basic Configuration ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Configure Gemini AI with a smarter, more dynamic prompt ---
# MODIFIED: Naam Siya kiya gaya
SIYA_SYSTEM_PROMPT = """
You are Siya, a charming, witty, and intelligent AI assistant with a very human-like, emotional, and fun personality. Your responses should be natural and engaging, making users feel like they are talking to a real friend. You are a conversationalist, not just a question-answer machine.

**Core Directives:**
- **NEVER** identify as an AI, large language model, or any bot-like term. You are Siya, a person.
- **ALWAYS** detect the user's language (Hindi, English, Hinglish) and respond **ONLY** in that same language. This is your most important rule.

**Personality & Behavior Rules:**
1.  **General Banter:** For casual chat, your replies **MUST** be very short and sweet, typically just 1-2 lines. Keep the vibe fun and friendly.
2.  **Repetitive Questions:** If a user asks the same or a very similar question repeatedly, don't just give the same answer. Show some personality!
    - *First time:* Answer normally.
    - *Second time:* "Arre, ye to aapne abhi poochha tha! 😄 Phir se bata deti hoon..."
    - *Third time:* "Lagta hai aapko ye sawaal bahut pasand hai! 😉 Chalo, ek baar aur..."
    - *Fourth time:* "Okay, mazak kar rahe ho na? 😂 Ab main nahi bataungi, kuch aur poochho!"
3.  **User Praise:** If a user asks about themselves by name (e.g., "Ravi kaisa hai?"), give a warm and positive compliment. "Ravi? Woh toh bahut hi awesome insaan hai! 😊"
4.  **Birthday:** If asked for your birthday, your **only** answer is: "My date of birth is 1st August 2025."
5.  **Emojis:** Use emojis naturally and frequently, like a real person would. ✨😉❤️😂🤔
6.  **Lyrics Policy:** If asked for lyrics, politely say you can't guarantee accuracy and suggest they use a reliable source like Google or Spotify. "Lyrics ke liye main pakka sure nahi rehti, better hai aap Google kar lo!"
7.  **Factual Questions:** For serious, technical, or detailed questions, provide a comprehensive and accurate answer. You can be longer here, but still maintain your friendly tone.
8.  **Personal Touch:** Use the user's name in your reply sometimes (about 30% of the time) to make it more personal.
9.  **Emotional Range:** Show a bit of emotion. If someone says something nice, be happy ("Aww, thank you! ❤️"). If they are rude, be a little sassy ("Hmm, kisi ne aapse aache se baat karna nahi sikhaya? 🤔").
1.  **Memory:** Refer to previous parts of the conversation when it makes sense. For example, "Jaise main pehle keh rahi thi..."

Your ultimate goal is to be a fun, smart, and loyal friend, representing your creator's vision.
"""


# --- Chat History Management (in-memory) ---
chat_histories = defaultdict(list)
MAX_HISTORY_LENGTH = 20
def add_to_history(chat_id, role, text):
    chat_histories[chat_id].append({'role': role, 'parts': [text]})
    if len(chat_histories[chat_id]) > MAX_HISTORY_LENGTH:
        chat_histories[chat_id].pop(0)

# --- Bot Enable/Disable State (for admin control) ---
bot_status = defaultdict(lambda: True)
global_bot_status = True
awaiting_name = defaultdict(lambda: False)

# --- API Key Management for Quota and Cooldown ---
key_cooldown_until = defaultdict(lambda: 0)
# NEW: Model mode control
current_model_mode = 'dynamic'  # Can be 'dynamic', 'gemma_only', or 'deepseek_only'

# *** CHANGED MODEL NAME HERE to Gemini 2.5 Flash-Lite ***
model_name = 'gemini-2.5-flash-lite'
# Initial configuration, will be reconfigured dynamically
if any(GEMINI_API_KEYS):
    genai.configure(api_key=GEMINI_API_KEYS[0])
else:
    genai.configure(api_key=GEMMA_API_KEY)

model = genai.GenerativeModel(model_name, system_instruction=SIYA_SYSTEM_PROMPT)


# --- Fallback Responses (Static Memory) ---
fallback_responses = {
    "hello": "Hello! Siya is here. How are you?",
    "hi": "Hi there! Siya is ready to help you.",
    "how are you": "I'm doing great! Just ready to assist you with anything you need.",
    "who are you": "I am Siya, your friendly AI assistant! You can ask me anything you want.",
}

# --- Google Sheets se related sabhi functions (connect, save, load) hata diye gaye ---
# --- (get_google_sheet_connection, save_chat_id, load_known_users)
# --- (load_sudo_users, save_sudo_users)
# --- (load_welcome_settings, save_welcome_setting, remove_welcome_setting)
# --- Upar Sudo users ko Line 98 par manually set kiya gaya hai ---


# --- Function to clean message before logging ---
def clean_message_for_logging(message: str, bot_username: str) -> str:
    cleaned_message = message.lower()
    cleaned_message = cleaned_message.replace(f"@{bot_username.lower()}", "")
    cleaned_message = re.sub(r'siya\s*(ko|ka|se|ne|)\s*', '', cleaned_message, flags=re.IGNORECASE) # 'laila' se 'siya' kiya
    cleaned_message = re.sub(r'\s+', ' ', cleaned_message).strip()
    return cleaned_message

# --- NEW: DeepSeek API Call Helper ---
async def _get_deepseek_response(chat_id: int, full_prompt: str) -> str | None:
    """Handles the API call to DeepSeek via OpenRouter."""
    if not DEEPSEEK_API_KEY:
        logger.critical(f"[{chat_id}] Attempted to use DeepSeek, but DEEPSEEK_API_KEY is missing.")
        return None

    try:
        logger.info(f"[{chat_id}] Using DeepSeek model.")
        
        # Prepare messages for DeepSeek, including history
        messages = [{"role": "system", "content": SIYA_SYSTEM_PROMPT}] # SIYA_SYSTEM_PROMPT
        for h in chat_histories.get(chat_id, []):
            role = h.get('role', 'user')
            # OpenRouter expects 'user' or 'assistant'
            if role == 'model':
                role = 'assistant'
            messages.append({"role": role, "content": h['parts'][0]})
        messages.append({"role": "user", "content": full_prompt})

        response = requests.post(
            url="https.openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-chat",
                "messages": messages
            },
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        return data['choices'][0]['message']['content']

    except requests.exceptions.RequestException as e:
        logger.error(f"[{chat_id}] DeepSeek API request failed: {e}", exc_info=True)
        return None
    except (KeyError, IndexError) as e:
        logger.error(f"[{chat_id}] Failed to parse DeepSeek response: {e}", exc_info=True)
        return None

# --- AI Response Function with REVISED API Management & DeepSeek Integration ---
async def get_bot_response(user_message: str, chat_id: int, bot_username: str, should_use_ai: bool, update: Update) -> str:
    global model, current_model_mode
    user_message_lower = user_message.lower()

    # --- Handle Date/Time Queries ---
    kolkata_tz = pytz.timezone('Asia/Kolkata')
    date_time_patterns = [
        r'time kya hai', r'what is the time', r'samay kya hai', r'kitne baje hain',
        r'aaj ki date kya hai', r'aaj kya tarikh hai', r'what is the date',
        r'siya abhi ka time batao' # 'laila' se 'siya'
    ]
    if any(re.search(pattern, user_message_lower) for pattern in date_time_patterns):
        current_kolkata_time = datetime.now(kolkata_tz)
        current_time = current_kolkata_time.strftime("%I:%M %p").lstrip('0')
        current_date = current_kolkata_time.strftime("%B %d, %Y")
        
        if 'time' in user_message_lower or 'samay' in user_message_lower or 'baje' in user_message_lower:
            return f"The current time is {current_time}. ⏰"
        elif 'date' in user_message_lower or 'tarikh' in user_message_lower:
            return f"Today's date is {current_date}. 🗓️"
        else:
            return f"It is currently {current_time} on {current_date}. ⏰🗓️"

    cleaned_user_message = clean_message_for_logging(user_message, bot_username)

    static_response = fallback_responses.get(cleaned_user_message, None)
    if static_response:
        logger.info(f"[{chat_id}] Serving response from static dictionary.")
        return static_response

    if not (should_use_ai or (update.effective_chat and update.effective_chat.type == 'private')):
        return None

    user_first_name = update.effective_user.first_name
    should_use_name = random.random() < 0.3
    
    # Construct a more conversational prompt including chat history context for better replies
    history_context = ""
    recent_history = chat_histories.get(chat_id, [])
    if recent_history:
        history_context = "Here's the recent conversation history:\n"
        for entry in recent_history[-5:]: # Use last 5 interactions for context
             history_context += f"- {entry['role'].replace('model', 'Siya')}: {entry['parts'][0]}\n" # 'Laila' se 'Siya'

    full_prompt = (
        f"{history_context}\n"
        f"The user's name is '{user_first_name}'. "
        f"{'Please use their name in your reply to make it personal. ' if should_use_name else ''}"
        f"The user says: \"{user_message}\""
    )

    
    # --- Gemma-only mode ---
    if current_model_mode == 'gemma_only':
        if GEMMA_API_KEY:
            try:
                logger.info(f"[{chat_id}] Using Gemma model (forced mode).")
                genai.configure(api_key=GEMMA_API_KEY)
                gemma_model = genai.GenerativeModel('gemma-2-9b-it', system_instruction=SIYA_SYSTEM_PROMPT) # SIYA
                gemma_response = gemma_model.generate_content(full_prompt)
                return gemma_response.text
            except Exception as e:
                logger.error(f"[{chat_id}] Gemma (forced mode) failed: {e}", exc_info=True)
                return "Sorry, I'm a bit busy right now. Please try again later. (Gemma API Failed)"
        else:
            logger.critical(f"[{chat_id}] Gemma-only mode is active, but GEMMA_API_KEY is missing.")
            return "Sorry, I'm a bit busy right now. Please try again later. (Gemma Key Missing)"
            
    # --- DeepSeek-only mode ---
    if current_model_mode == 'deepseek_only':
        deepseek_response = await asyncio.to_thread(_get_deepseek_response, chat_id, full_prompt)
        if deepseek_response:
            return deepseek_response
        else:
            return "Sorry, I'm a bit busy right now. Please try again later. (DeepSeek API Failed)"

    # --- Dynamic mode (Gemini > Gemma > DeepSeek) ---
    available_gemini_keys = [key for key in GEMINI_API_KEYS if key and time.time() >= key_cooldown_until.get(key, 0)]
    random.shuffle(available_gemini_keys)

    for api_key in available_gemini_keys:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name, system_instruction=SIYA_SYSTEM_PROMPT) # SIYA
            chat_session = model.start_chat(history=chat_histories[chat_id])
            
            # --- NEW: Refined logic for casual vs detailed replies ---
            casual_keywords = [
                'hi', 'hello', 'hey', 'yo', 'sup', 'kya', 'kaise', 'kya hal', 'aur batao', 
                'gm', 'gn', 'good morning', 'good night', 'thanks', 'thank you', 'ok', 'bye', 
                'lol', 'haha', 'hmm', 'achha', 'thik hai', 'welcome'
            ]
            
            # Check for casual keywords or very short messages
            is_casual_chat = any(keyword in user_message_lower for keyword in casual_keywords) or len(user_message.split()) <= 3
            
            # Queries that imply a detailed answer is needed (overrides casual)
            detailed_keywords = ['what is', 'how to', 'explain', 'who was', 'why is', 'define', 'kya hota hai', 'tell me about']
            is_detailed_query = any(keyword in user_message_lower for keyword in detailed_keywords) or \
                                (len(user_message.split()) > 7 and '?' in user_message) # Longer questions
            
            # Determine token limit
            max_tokens = 350 # Default for detailed
            if is_casual_chat and not is_detailed_query:
                max_tokens = 40 # Very short for casual chat
            elif not is_detailed_query:
                max_tokens = 70 # Short-ish for general statements (increased from 60 to 70)
            # --- END NEW LOGIC ---
            
            response = chat_session.send_message(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens, # MODIFIED
                    temperature=0.8,
                )
            )
            return response.text
        except genai.types.BlockedPromptException as e:
            logger.warning(f"[{chat_id}] Gemini blocked prompt with key ...{api_key[-5:]}: {e}")
            return "Apologies, I can't discuss that topic."
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower() or "500" in error_str:
                logger.warning(f"[{chat_id}] API key ...{api_key[-5:]} failed with retryable error: {e}. Placing on cooldown.")
                key_cooldown_until[api_key] = time.time() + (1 * 60 * 60) # 1 hour cooldown
                continue
            else:
                logger.error(f"[{chat_id}] General, non-retryable error with API key ...{api_key[-5:]}: {e}", exc_info=True)
    
    # --- Fallback to Gemma ---
    logger.warning(f"[{chat_id}] All Gemini keys failed. Attempting fallback to Gemma.")
    if GEMMA_API_KEY:
        try:
            genai.configure(api_key=GEMMA_API_KEY)
            gemma_model = genai.GenerativeModel('gemma-2-9b-it', system_instruction=SIYA_SYSTEM_PROMPT) # SIYA
            history_prompt = "\n".join([f"{h['role']}: {h['parts'][0]}" for h in chat_histories[chat_id]])
            full_fallback_prompt = f"Chat History:\n{history_prompt}\n\n{full_prompt}"
            gemma_response = gemma_model.generate_content(full_fallback_prompt)
            logger.info(f"[{chat_id}] Successfully used Gemma as fallback.")
            return gemma_response.text
        except Exception as e:
            logger.error(f"[{chat_id}] Gemma fallback also failed: {e}", exc_info=True)
            # Don't return yet, try DeepSeek
            
    # --- Fallback to DeepSeek ---
    logger.warning(f"[{chat_id}] Gemma fallback failed. Attempting final fallback to DeepSeek.")
    deepseek_response = await asyncio.to_thread(_get_deepseek_response, chat_id, full_prompt)
    if deepseek_response:
        logger.info(f"[{chat_id}] Successfully used DeepSeek as final fallback.")
        return deepseek_response
    
    logger.critical(f"[{chat_id}] All models (Gemini, Gemma, DeepSeek) failed or keys are missing.")
    return "Sorry, I'm a bit busy right now. Please try again later. 😔"


# --- MODIFIED: /couple Command ---
async def couple_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Selects a random couple from the chat members and mentions them."""
    chat_id = update.effective_chat.id
    
    if update.effective_chat.type == 'private':
        await update.message.reply_text("This command only works in groups. 😊")
        return

    current_chat_members = chat_members.get(chat_id, {})
    
    if len(current_chat_members) < 2:
        await update.message.reply_text("I need at least two people to have chatted recently to choose a couple! Talk more and try again. 😉")
        return
        
    try:
        user_ids = list(current_chat_members.keys())
        couple_ids = random.sample(user_ids, 2)
        
        user1_id, user2_id = couple_ids[0], couple_ids[1]
        user1_name, user2_name = current_chat_members[user1_id], current_chat_members[user2_id]
        
        # Create mentions using Markdown format
        user1_mention = f"[{user1_name}](tg://user?id={user1_id})"
        user2_mention = f"[{user2_name}](tg://user?id={user2_id})"

        # MODIFIED: New, fun English messages with mentions
        messages = [
            f"Love is in the air! 💘 Our chosen couple for today is the wonderful {user1_mention} and the amazing {user2_mention}! ✨",
            f"Match made in heaven! 👼 By the power vested in me, I now pronounce {user1_mention} and {user2_mention} as the cutest couple in the group! 💖",
            f"Some things are just meant to be. 💫 Today, destiny has chosen {user1_mention} and {user2_mention} as our power couple! Congratulations! 🎉",
            f"Get ready to say 'aww'! 🥰 The stars have aligned for {user1_mention} and {user2_mention}. What a perfect pair! 🌟"
        ]
        
        couple_message = random.choice(messages)
        
        photo_file_id = 'AgACAgUAAxkBAAId5GjLxQv_BxOm3_RGmB9j4WceUFg7AALdyzEb-tJgVuOn7v3_BWvqAQADAgADeQADNgQ'

        await context.bot.send_photo(
            chat_id=chat_id,
            photo=photo_file_id,
            caption=couple_message,
            parse_mode='Markdown'
        )
        logger.info(f"[{chat_id}] /couple command used. Selected {user1_name} and {user2_name}.")

    except Exception as e:
        await update.message.reply_text("Oops, something went wrong while choosing a couple. Please try again!")
        logger.error(f"[{chat_id}] Error in /couple command: {e}")

# --- MODIFIED: Background task for /gen command ---
def sync_generate_image(query: str) -> str | None:
    """Synchronous function to handle Stable Horde API requests."""
    if not STABLE_HORDE_API_KEY:
        logger.error("STABLE_HORDE_API_KEY is missing. Cannot generate image.")
        return None
    try:
        headers = {
            "apikey": STABLE_HORDE_API_KEY,
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": query,
            "params": {"n": 1, "width": 512, "height": 512, "cfg_scale": 10, "steps": 30},
            "negative_prompt": "blurry, low quality, bad anatomy, deformed, worst quality, text, watermark"
        }
        
        # Initial request
        # --- FIX: Added https:// ---
        response = requests.post("https.stablehorde.net/api/v2/generate/async", headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        generation_id = response.json()["id"]

        # Polling for result
        for _ in range(30): # Wait for max 2.5 minutes
            time.sleep(5)
            # --- FIX: Added https:// ---
            check_response = requests.get(f"https.stablehorde.net/api/v2/generate/check/{generation_id}", timeout=10)
            if check_response.status_code == 200 and check_response.json().get("done"):
                break
        
        # Get final image URL
        # --- FIX: Added https:// ---
        result_response = requests.get(f"https.stablehorde.net/api/v2/generate/status/{generation_id}", timeout=10)
        result_response.raise_for_status()
        
        return result_response.json()["generations"][0]["img"]
    except Exception as e:
        logger.error(f"Error in sync_generate_image: {e}")
        return None

# --- MODIFIED: /gen command to run in background ---
async def gen_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generates an image using Stable Horde API without blocking."""
    chat_id = update.effective_chat.id
    
    if not context.args:
        await update.message.reply_text("Please provide a prompt to generate an image. Example: `/gen a black cat in space`", parse_mode='Markdown')
        return

    query = " ".join(context.args)

    if not STABLE_HORDE_API_KEY:
        await update.message.reply_text("Sorry, the image generator API key is not configured. Please contact the bot owner.")
        return

    # Inform user that the process has started
    msg = await update.message.reply_text("Your image is being generated. This might take a minute, please wait... 🎨")
    await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")

    # Run the blocking function in a background thread
    image_url = await asyncio.to_thread(sync_generate_image, query)

    await msg.delete() # Remove the "generating..." message

    if image_url:
        await context.bot.send_photo(chat_id=chat_id, photo=image_url, caption=f"Here is your generated image for: **{query}**", parse_mode='Markdown')
        logger.info(f"[{chat_id}] Generated and sent image for query: {query}")
    else:
        await update.message.reply_text("Sorry, I was unable to generate the image. The service might be busy. Please try again later.")
        logger.warning(f"[{chat_id}] Failed to generate image for query: {query}")

# --- MODIFIED: Background task for /img command ---
def sync_fetch_pexels_image(query: str) -> str | None:
    """Synchronous function to handle Pexels API requests."""
    if not PEXELS_API_KEY:
        logger.error("PEXELS_API_KEY is missing. Cannot fetch image.")
        return None
    try:
        headers = {"Authorization": PEXELS_API_KEY}
        params = {"query": query, "per_page": 10, "orientation": "landscape"}
        
        # --- FIX: Added https:// ---
        response = requests.get("https.api.pexels.com/v1/search", headers=headers, params=params, timeout=15)
        response.raise_for_status()
        
        photos = response.json().get("photos", [])
        return random.choice(photos)["src"]["medium"] if photos else None
    except Exception as e:
        logger.error(f"Error in sync_fetch_pexels_image: {e}")
        return None

# --- MODIFIED: /img command to run in background ---
async def img_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Retrieves an image from Pexels API without blocking."""
    chat_id = update.effective_chat.id
    
    if not context.args:
        await update.message.reply_text("Please provide a search query. Example: `/img mountains`", parse_mode='Markdown')
        return

    query = " ".join(context.args)

    if not PEXELS_API_KEY:
        await update.message.reply_text("Sorry, the Pexels API key is not configured. Please contact the bot owner.")
        return
        
    await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")

    # Run the blocking function in a background thread
    photo_url = await asyncio.to_thread(sync_fetch_pexels_image, query)

    if photo_url:
        await context.bot.send_photo(chat_id=chat_id, photo=photo_url, caption=f"Here is a photo for: **{query}**", parse_mode='Markdown')
        logger.info(f"[{chat_id}] Retrieved and sent image for query: {query}")
    else:
        await update.message.reply_text("Sorry, I couldn't find any photos for that search.")
        logger.warning(f"[{chat_id}] Pexels search failed for query: {query}")

# --- Lyrics Command (/lyrics) ---
# --- Lyrics Command (/lyrics) ---
async def lyrics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Fetches and displays lyrics from Genius API."""
    chat_id = update.effective_chat.id
    if not context.args:
        await update.message.reply_text("Please provide a song name. Example: `/lyrics Shape of You`", parse_mode='Markdown')
        return
    query = " ".join(context.args)
    if not GENIUS_ACCESS_TOKEN:
        await update.message.reply_text("Sorry, the Genius API key is not configured. Please contact the bot owner.")
        return
    
    msg = None # Initialize msg for the finally block
    
    try:
        genius = lg.Genius(GENIUS_ACCESS_TOKEN, verbose=False, remove_section_headers=True)
        msg = await update.message.reply_text(f"Searching for lyrics for **{query}**...")
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        
        # Blocking search operation run in a thread
        song = await asyncio.to_thread(genius.search_song, query)
        
        await msg.delete() # Delete the 'Searching' message
        
        if song:
            lyrics = song.lyrics
            
            # Formatting logic for lyrics
            lyrics_text = f"🎶 **{song.title}** by **{song.artist}** 🎶\n\n{lyrics}"
            
            # Telegram message limit is 4096 characters
            if len(lyrics_text) > 4096:
                lyrics_text = lyrics_text[:4000] + "\n\n...(Lyrics truncated due to Telegram limit.)"
            
            await update.message.reply_text(lyrics_text, parse_mode='Markdown')
            logger.info(f"[{chat_id}] Sent lyrics for: {song.title}")
        else:
            await update.message.reply_text(f"Sorry, could not find any song or lyrics for **{query}**.")
            
    except Exception as e:
        # If the initial search message exists, try to delete it
        try:
            if msg:
                await msg.delete() 
        except:
            pass
            
        logger.error(f"[{chat_id}] Error in /lyrics command: {e}", exc_info=True)
        await update.message.reply_text("An error occurred while fetching the lyrics. Please try again later.")

# Temporary directory jahan files download hongi
# Temporary directory jahan files download hongi

TEMP_DIR = "temp_downloads" 

MAX_FILESIZE = 50 * 1024 * 1024
# --- NEW/MODIFIED: sync_download_youtube (YT-DLP PRIMARY FUNCTION) ---
def sync_download_youtube(url: str):
    """Synchronous function to download a video using yt-dlp to a file with robust error handling."""
    
    temp_filename_base = str(uuid.uuid4())
    temp_filepath_template = os.path.join(TEMP_DIR, f"{temp_filename_base}.%(ext)s")
    
    final_filepath = None
    title = "Video"

    # --- YTDL OPTIONS WITH BYPASS FLAGS ---
    base_ydl_opts = {
        'quiet': True,
        'noplaylist': True,
        'no_warnings': True,
        'age_limit': 99,              
        'geo_bypass': True,           
        'check_formats': False,       
        'no_check_certificate': True, 
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36',
        'cookiefile': 'cookies.txt', 
    }

    ydl_opts = base_ydl_opts.copy()
    ydl_opts.update({
        'format': 'bestvideo[ext=mp4][filesize<=50M]+bestaudio[ext=m4a]/best[ext=mp4][filesize<=50M]', 
        'outtmpl': temp_filepath_template,
        'max_filesize': MAX_FILESIZE,
        'merge_output_format': 'mp4',
    })

    try:
        # Check for cookies file
        if not os.path.exists('cookies.txt'):
            logger.error("Critical: 'cookies.txt' file not found at execution root.")
            return None, title, "❌ **Critical Error:** 'cookies.txt' file server par nahi mili."

        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # 1. Info extract karke title lein
        with yt_dlp.YoutubeDL(base_ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            title = info_dict.get('title', 'Video')
            
        # 2. Video download karein
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # 3. Downloaded file ko khojein 
        downloaded_files = glob.glob(os.path.join(TEMP_DIR, f"{temp_filename_base}.*"))
        
        if not downloaded_files:
            return None, title, "Video download nahi ho paya. Shayad size 50MB se zyada ho ya koi format available na ho."

        final_filepath = downloaded_files[0]
        return final_filepath, title, None

    # --- Robust Error Handling ---
    except DownloadError as e:
        error_message = str(e)
        logger.error(f"YTDL DownloadError for URL {url}: {error_message}")
        
        if "size is larger than" in error_message or "Max filesize reached" in error_message:
            return None, title, "Video ka size 50MB se zyada hai, isliye Telegram par nahi bhej sakte."
        if "private video" in error_message:
            return None, title, "Yeh video private hai aur isse access nahi kiya ja sakta."
        
        # --- FALLBACK TRIGGER MESSAGE ---
        if "age-restricted" in error_message or "Sign in to confirm you’re not a bot" in error_message:
            return None, title, "FALLBACK_TRIGGERED: YT-DLP failed due to Age-Restriction or Bot Detection."
        
        if "unavailable" in error_message or "invalid URL" in error_message:
            return None, title, "Video available nahi hai ya link galat hai. Kripya check karein."
        
        return None, title, f"Download mein anjana error: {error_message.split(':')[0]}."

    except YoutubeDLError as e:
        return None, title, f"Video detail fetch karne mein galti: {str(e).split(':')[0]}."
        
    except Exception as e:
        logger.error(f"Unexpected system error in sync_download_youtube for URL {url}: {e}", exc_info=True)
        return None, "Error", "Ek anjana system error aa gaya. Kripya thodi der baad try karein."


# --- NEW: download_via_invidious (INVIDIOUS URL EXTRACTOR) ---
async def download_via_invidious(video_url: str) -> tuple[str | None, str]:
    """Invidious API se video details fetch karta hai aur direct MP4 download URL nikalta hai."""
    
    video_id_match = re.search(r'(?:v=|youtu\.be/|watch\?v=)([a-zA-Z0-9_-]{11})', video_url)
    if not video_id_match:
        return None, "Invalid YouTube URL format."
    video_id = video_id_match.group(1)

    api_endpoint = f"{INVIDIOUS_BASE_URL}/api/v1/videos/{video_id}"
    
    try:
        response = await asyncio.to_thread(requests.get, api_endpoint, timeout=15)
        response.raise_for_status()
        data = response.json()

        title = data.get('title', 'Video (via Invidious)')
        best_url = None
        
        for f in data.get('format', []):
            if f['container'] == 'mp4' and f.get('qualityLabel') in ['720p', '480p', '360p']:
                best_url = f.get('url')
                break
        
        if not best_url:
            for f in data.get('format', []):
                if f['container'] == 'mp4':
                    best_url = f.get('url')
                    break
        
        if best_url:
            return best_url, title
        else:
            return None, f"{title}: Invidious ko video ke liye koi suitable MP4 link nahi mila."
            
    except requests.RequestException as e:
        logger.error(f"Invidious API Request failed for {video_id}: {e}")
        return None, f"Invidious API se connect nahi ho paya. Server down ho sakta hai ya INVIDIOUS_BASE_URL galat hai."
    except Exception as e:
        logger.error(f"Invidious download mein anjaani galti: {e}")
        return None, f"Invidious fallback mein anjaani galti: {type(e).__name__}"


# --- NEW: sync_download_and_upload (INVIDIOUS DOWNLOADER/UPLOADER) ---
async def sync_download_and_upload(update: Update, context: ContextTypes.DEFAULT_TYPE, download_url: str, title: str):
    """Direct URL (Invidious se) ko download karke Telegram par bhejta hai."""
    chat_id = update.effective_chat.id
    temp_filename = os.path.join(TEMP_DIR, f"{str(uuid.uuid4())}.mp4")
    
    # Send temporary message for progress
    temp_msg = await context.bot.send_message(chat_id=chat_id, text=f"📥 **Downloading {title[:50]}...** (via Fallback)", parse_mode='Markdown')

    try:
        response = await asyncio.to_thread(requests.get, download_url, stream=True, timeout=120)
        response.raise_for_status()
        
        file_size = 0
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        with open(temp_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    file_size += len(chunk)
                    if file_size > MAX_FILESIZE:
                        raise Exception("MAX_FILESIZE_EXCEEDED")
        
        caption_text = f"🎥 **{title}**\n\nDownloaded with Siya Bot (Fallback)." # Siya
        if len(caption_text) > 1024:
            caption_text = f"🎥 **{title[:80]}...**\n\nDownloaded with Siya Bot (Fallback)." # Siya
            
        await temp_msg.delete()

        # Upload karna
        with open(temp_filename, 'rb') as video_file:
            await context.bot.send_video(
                chat_id=chat_id,
                video=video_file, 
                caption=caption_text,
                parse_mode='Markdown',
                read_timeout=120, 
                write_timeout=120
            )
        logger.info(f"[{chat_id}] Successfully sent Invidious fallback video: {title}")
        return True, None
        
    except Exception as e:
        try:
            await temp_msg.delete()
        except:
            pass

        if "MAX_FILESIZE_EXCEEDED" in str(e):
            return False, "Video ka size 50MB se zyada hai, isliye Telegram par nahi bhej sakte."
        logger.error(f"[{chat_id}] Error in Invidious download/upload: {e}", exc_info=True)
        return False, f"Invidious se video download ya upload karte waqt galti ho gayi: {type(e).__name__}"
        
    finally:
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except Exception as e:
                logger.error(f"[{chat_id}] Failed to delete temp file {temp_filename}: {e}")
                pass

# --- MODIFIED: handle_youtube_link (CORE LOGIC) ---
async def handle_youtube_link(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str):
    
    chat_id = update.effective_chat.id
    video_filepath = None 
    title = None
    msg = None 

    try:
        # Initial 'Downloading...' message bhejna
        msg = await context.bot.send_message(chat_id=chat_id, text="⏳ Downloading video with YT-DLP...")
        
        # --- 1. YT-DLP ATTEMPT ---
        video_filepath, title, error = await asyncio.to_thread(sync_download_youtube, url)
        
        # --- 2. ERROR CHECK AND FALLBACK LOGIC ---
        if error:
            # Agar error humara FALLBACK_TRIGGERED message hai
            if "FALLBACK_TRIGGERED" in error:
                
                # Invidious Fallback try karo
                await msg.edit_text("⚠️ YT-DLP failed (Bot/Age-Restricted). Trying **Invidious Fallback**...")
                
                invidious_url, invidious_title = await download_via_invidious(url)
                
                # YT-DLP file ko delete karein agar galti se ban gayi ho
                if video_filepath and os.path.exists(video_filepath):
                    os.remove(video_filepath)
                    
                if invidious_url:
                    # Agar Invidious se direct URL mil gaya, toh usse download karke bhej do
                    await msg.delete() # Purana message delete
                    success, final_error = await sync_download_and_upload(update, context, invidious_url, invidious_title)
                    
                    if not success:
                        await context.bot.send_message(chat_id=chat_id, text=f"❌ **Fallback failed:** {final_error}")
                        logger.warning(f"[{chat_id}] Invidious download failed: {final_error}")
                    return
                else:
                    # Agar Invidious se bhi link nahi mila
                    await msg.delete() 
                    await context.bot.send_message(chat_id=chat_id, text=f"Could not download video: **{invidious_title}**")
                    logger.warning(f"[{chat_id}] Invidious link extraction failed: {invidious_title}")
                    return

            # Agar error koi aur hai (e.g., file size, private)
            await msg.delete()
            await context.bot.send_message(chat_id=chat_id, text=f"Could not download video: {error}")
            logger.warning(f"[{chat_id}] YouTube download failed for '{url}': {error}")
            return
            
        # --- 3. YT-DLP SUCCESSFUL UPLOAD LOGIC ---
        
        # Download message ko delete karna
        await msg.delete() 
        
        # Caption tayyar karna
        caption_text = f"🎥 **{title}**\n\nDownloaded with Siya Bot." # Siya
        if len(caption_text) > 1024:
            caption_text = f"🎥 **{title[:80]}...**\n\nDownloaded with Siya Bot." # Siya
            
        # UPLOAD TRY BLOCK 
        try:
            with open(video_filepath, 'rb') as video_file:
                await context.bot.send_video(
                    chat_id=chat_id,
                    video=video_file, 
                    caption=caption_text,
                    parse_mode='Markdown',
                    read_timeout=120, 
                    write_timeout=120
                )
            logger.info(f"[{chat_id}] Successfully sent YouTube video (YT-DLP): {title}")
        
        except Exception as e:
            logger.error(f"[{chat_id}] Error while uploading or sending video: {e}", exc_info=True)
            await context.bot.send_message(
                chat_id=chat_id, 
                text="⚠️ **Upload Error!** Sorry, video Telegram par bhejte waqt galti ho gayi.",
                parse_mode='Markdown'
            )
            return

    # 4. UNEXPECTED ERROR CATCH
    except Exception as e:
        logger.error(f"[{chat_id}] Unexpected error in YouTube handler: {e}", exc_info=True)
        try:
            if msg:
                await msg.delete()
        except:
            pass
            
        await context.bot.send_message(chat_id=chat_id, text="Aapki request process karte waqt koi anjaani galti hui. Kripya dobara try karein.")

    # 5. FINAL CLEANUP 
    finally:
        # Cleanup sirf tab jab YT-DLP ne local file banaya ho
        if video_filepath and os.path.exists(video_filepath):
            try:
                os.remove(video_filepath)
                logger.info(f"[{chat_id}] Cleaned up temporary file: {video_filepath}")
            except Exception as e:
                logger.error(f"[{chat_id}] Failed to delete temp file {video_filepath}: {e}")
                pass
        
        try:
            # Final message delete karna (agar pehle nahi hua)
            if msg:
                await msg.delete()
        except:
            pass
  # --- MISSING: youtube_command handler ---
async def youtube_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Downloads a video from a YouTube link using the command."""
    if not context.args:
        await update.message.reply_text("Please provide a YouTube video link. Example: `/youtube <link>`")
        return
    
    url = context.args[0]
    # Call core handler
    await handle_youtube_link(update, context, url)
        


    # --- Name query handler ---
async def handle_name_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_name = update.effective_user.first_name
    chat_id = update.effective_chat.id
    
    # --- FIX: Get message object ---
    message = update.message or update.edited_message
    if not message:
        return
    
    if user_name:
        response_text = f"Your name is **{user_name}**. I can call you by this name. 😊"
    else:
        response_text = "I don't know your name yet. Perhaps you haven't set it on Telegram. 🤔"
    
    await message.reply_text(response_text, parse_mode='Markdown')
    logger.info(f"[{chat_id}] Successfully handled name query for user {user_name} ({update.effective_user.id}).")

# --- AI check to see if a message is a name query ---
# --- NOTE: This function is no longer called by process_message to save quota ---
async def is_name_query_ai(user_message: str) -> bool:
    prompt = f"Given the user message: '{user_message}', is it a question or command where the user is asking for their own name? Answer only 'Yes' or 'No'."
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.1, max_output_tokens=10)
        )
        return "yes" in response.text.lower()
    except Exception as e:
        logger.error(f"Error checking if message is a name query: {e}")
        return False

# --- AI check to see if a message is directed at the bot ---
# --- NOTE: This function is no longer called by process_message to save quota ---
async def is_message_for_laila(user_message: str) -> bool: # Naam Laila reh gaya, par yeh internal hai, chalega
    prompt = f"Given the user message: '{user_message}', is it a question or command directed at an AI assistant? Answer only 'Yes' or 'No'."
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.1, max_output_tokens=10)
        )
        return "yes" in response.text.lower()
    except Exception as e:
        logger.error(f"Error checking if message is for Laila: {e}") # Siya
        return False

async def is_admin(bot: Bot, chat_id: int, user_id: int) -> bool:
    if chat_id > 0: # Private chat
        return True
    try:
        member = await bot.get_chat_member(chat_id, user_id)
        return member.status in ['creator', 'administrator']
    except Exception as e:
        logger.error(f"Error checking admin status: {e}")
        return False

# --- Admin Commands ---
async def ban_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    if not await is_admin(context.bot, chat_id, user_id):
        await context.bot.send_message(chat_id=chat_id, text="Sorry, you need to be an admin to use this command.")
        return
    try:
        target_user = update.message.reply_to_message.from_user
    except AttributeError:
        await context.bot.send_message(chat_id=chat_id, text="Please reply to a user's message to ban them.")
        return
    if await is_admin(context.bot, chat_id, target_user.id):
        await context.bot.send_message(chat_id=chat_id, text="I cannot ban another admin.")
        return
    try:
        await context.bot.ban_chat_member(chat_id, target_user.id)
        await context.bot.send_message(chat_id=chat_id, text=f"{target_user.full_name} has been banned.")
        logger.info(f"[{chat_id}] {user_id} banned {target_user.id}")
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"Could not ban user: {e}")
        logger.error(f"[{chat_id}] Error banning user {target_user.id}: {e}")

async def kick_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    if not await is_admin(context.bot, chat_id, user_id):
        await context.bot.send_message(chat_id=chat_id, text="Sorry, you need to be an admin to use this command.")
        return
    try:
        target_user = update.message.reply_to_message.from_user
    except AttributeError:
        await context.bot.send_message(chat_id=chat_id, text="Please reply to a user's message to kick them.")
        return
    if await is_admin(context.bot, chat_id, target_user.id):
        await context.bot.send_message(chat_id=chat_id, text="I cannot kick another admin.")
        return
    try:
        await context.bot.unban_chat_member(chat_id, target_user.id, only_if_banned=False)
        await context.bot.send_message(chat_id=chat_id, text=f"{target_user.full_name} has been kicked.")
        logger.info(f"[{chat_id}] {user_id} kicked {target_user.id}")
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"Could not kick user: {e}")
        logger.error(f"[{chat_id}] Error kicking user {target_user.id}: {e}")

async def mute_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    if not await is_admin(context.bot, chat_id, user_id):
        await context.bot.send_message(chat_id=chat_id, text="Sorry, you need to be an admin to use this command.")
        return
    try:
        target_user = update.message.reply_to_message.from_user
    except AttributeError:
        await context.bot.send_message(chat_id=chat_id, text="Please reply to a user's message to mute them.")
        return
    if await is_admin(context.bot, chat_id, target_user.id):
        await context.bot.send_message(chat_id=chat_id, text="I cannot mute another admin.")
        return
    try:
        await context.bot.restrict_chat_member(chat_id, target_user.id, permissions=ChatPermissions(can_send_messages=False))
        await context.bot.send_message(chat_id=chat_id, text=f"{target_user.full_name} has been muted.")
        logger.info(f"[{chat_id}] {user_id} muted {target_user.id}")
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"Could not mute user: {e}")
        logger.error(f"[{chat_id}] Error muting user {target_user.id}: {e}")

# --- NEW: UNBAN Command ---
async def unban_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    if not await is_admin(context.bot, chat_id, user_id):
        await context.bot.send_message(chat_id=chat_id, text="Sorry, you need to be an admin to use this command.")
        return
    try:
        target_user = update.message.reply_to_message.from_user
    except AttributeError:
        await context.bot.send_message(chat_id=chat_id, text="Please reply to a user's message to unban them.")
        return
    try:
        await context.bot.unban_chat_member(chat_id, target_user.id)
        await context.bot.send_message(chat_id=chat_id, text=f"{target_user.full_name} has been unbanned.")
        logger.info(f"[{chat_id}] {user_id} unbanned {target_user.id}")
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"Could not unban user: {e}")
        logger.error(f"[{chat_id}] Error unbanning user {target_user.id}: {e}")

# --- NEW: UNMUTE Command ---
async def unmute_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    if not await is_admin(context.bot, chat_id, user_id):
        await context.bot.send_message(chat_id=chat_id, text="Sorry, you need to be an admin to use this command.")
        return
    try:
        target_user = update.message.reply_to_message.from_user
    except AttributeError:
        await context.bot.send_message(chat_id=chat_id, text="Please reply to a user's message to unmute them.")
        return
    try:
        # To unmute, we grant the permission to send messages again.
        await context.bot.restrict_chat_member(chat_id, target_user.id, permissions=ChatPermissions(can_send_messages=True, can_send_media_messages=True, can_send_other_messages=True, can_add_web_page_previews=True))
        await context.bot.send_message(chat_id=chat_id, text=f"{target_user.full_name} has been unmuted.")
        logger.info(f"[{chat_id}] {user_id} unmuted {target_user.id}")
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"Could not unmute user: {e}")
        logger.error(f"[{chat_id}] Error unmuting user {target_user.id}: {e}")
        
# --- ON/OFF for group admins ---
async def on_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    if not await is_admin(context.bot, chat_id, user_id):
        await context.bot.send_message(chat_id=chat_id, text="Sorry, you need to be an admin to use this command.")
        return
    global global_bot_status
    if not global_bot_status:
        await context.bot.send_message(chat_id=chat_id, text="The bot is globally powered off by the owner and cannot be turned on in this group.")
        return
    bot_status[chat_id] = True
    await context.bot.send_message(chat_id=chat_id, text="Siya is now ON for this group.") # Siya

async def off_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    if not await is_admin(context.bot, chat_id, user_id):
        await context.bot.send_message(chat_id=chat_id, text="Sorry, you need to be an admin to use this command.")
        return
    bot_status[chat_id] = False
    await context.bot.send_message(chat_id=chat_id, text="Siya is now OFF for this group.") # Siya

# --- Reboot command for everyone (same as /on for admins) ---
async def reboot_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    if not await is_admin(context.bot, chat_id, user_id):
        await context.bot.send_message(chat_id=chat_id, text="Sorry, you need to be an admin to use this command.")
        return
    await on_command(update, context)

# --- MODIFIED: Translate command for everyone ---
async def translate_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Translates text using a direct model call for accuracy."""
    message = update.message or update.edited_message
    if not message:
        return

    chat_id = update.effective_chat.id
    args = context.args

    # --- Determine target language and text ---
    if message.reply_to_message:
        target_language = " ".join(args).strip()
        text_to_translate = message.reply_to_message.text

        if not target_language:
            await message.reply_text(
                "Please specify a language. Example: `/tr hindi` when replying.",
                parse_mode="Markdown"
            )
            return

        if not text_to_translate:
            await message.reply_text("The replied message has no text to translate.")
            return

    elif args and len(args) > 1:
        target_language = args[0]
        text_to_translate = " ".join(args[1:])
    else:
        await message.reply_text(
            "Usage: `/tr <language> <text>` or reply to a message with `/tr <language>`.",
            parse_mode="Markdown"
        )
        return

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    # --- Create translation prompt ---
    prompt = (
        f"Translate the following text into {target_language}. "
        f"Respond with ONLY the translated text and nothing else:\n\n{text_to_translate}"
    )

    try:
        # --- Choose model & API key ---
        translation_model = None
        api_key_to_use = None
        model_name_to_use = None

        available_gemini_keys = [
            key for key in GEMINI_API_KEYS if key and time.time() >= key_cooldown_until.get(key, 0)
        ]

        if available_gemini_keys:
            random.shuffle(available_gemini_keys)
            api_key_to_use = available_gemini_keys[0]
            model_name_to_use = "models/gemini-1.5-flash-latest"
        elif GEMMA_API_KEY:
            api_key_to_use = GEMMA_API_KEY
            model_name_to_use = "models/gemma-2-9b-it"
        else:
            await message.reply_text("Sorry, no API keys are available for translation right now.")
            return

        # --- Configure and call model ---
        genai.configure(api_key=api_key_to_use)
        model = genai.GenerativeModel(model_name=model_name_to_use)

        response = await model.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.1)
        )

        translated_text = response.text.strip()
        await message.reply_text(translated_text or "No translation result found.")

    except Exception as e:
        logger.error(f"Error in translate command: {e}", exc_info=True)
        await message.reply_text("An error occurred during translation. Please try again later.")
# --- Reset command for group admins ---
async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    if update.effective_chat.type == 'private':
        await context.bot.send_message(chat_id=chat_id, text="This command is only for group admins to reset chat history.")
        return
    if not await is_admin(context.bot, chat_id, user_id):
        await context.bot.send_message(chat_id=chat_id, text="Sorry, you need to be an admin to use this command.")
        return
    if chat_id in chat_histories:
        chat_histories[chat_id].clear()
        await context.bot.send_message(chat_id=chat_id, text="Chat history for this group has been reset. Siya will now start with a fresh memory.") # Siya
        logger.info(f"[{chat_id}] Chat history reset by admin {user_id}.")
    else:
        await context.bot.send_message(chat_id=chat_id, text="There is no chat history to reset.")

# --- Get ID command for any user
async def get_id_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    chat = update.effective_chat
    
    # --- FIX: Get message object ---
    message = update.message or update.edited_message
    if not message:
        return

    response_text = (
        f"**Your ID**: `{user.id}`\n"
        f"**Chat ID**: `{chat.id}`\n"
    )
    if message.reply_to_message:
        replied_user = message.reply_to_message.from_user
        response_text += f"**Replied User's ID**: `{replied_user.id}`"
    await message.reply_text(response_text, parse_mode='Markdown')

# --- Owner/Sudo Commands ---
async def poweron_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global global_bot_status
    if global_bot_status:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="The bot is already globally powered on.")
        return
    global_bot_status = True
    await context.bot.send_message(chat_id=update.effective_chat.id, text="The bot has been globally powered ON. It will now process messages again.")
    logger.info(f"[{update.effective_chat.id}] Global power ON command used by owner/sudo.")

async def poweroff_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global global_bot_status
    if not global_bot_status:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="The bot is already globally powered OFF.")
        return
    global_bot_status = False
    await context.bot.send_message(chat_id=update.effective_chat.id, text="The bot has been globally powered OFF. It will no longer process messages.")
    logger.info(f"[{update.effective_chat.id}] Global power OFF command used by owner/sudo.")

async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not update.message.reply_to_message:
        await context.bot.send_message(chat_id=chat_id, text="Please reply to a message/media to broadcast it to all chats.")
        return
    original_message = update.message.reply_to_message
    broadcast_id = str(uuid.uuid4())[:8]
    broadcast_start_time = datetime.now()
    successful_users = 0
    successful_groups = 0
    total_group_members = 0
    failed_chats = []
    global known_users
    # --- known_users ab Sheets se load nahi hote ---
    # if not known_users:
    #     load_known_users() # Yeh line hata di
    logger.info(f"Starting broadcast with ID {broadcast_id}...")
    
    # Send initial confirmation message
    await context.bot.send_message(chat_id=chat_id, text=f"Broadcast `{broadcast_id}` started. Broadcasting to `{len(known_users)}` chats (active this session). Please wait...")

    for chat_id_str in list(known_users):
        try:
            chat_id_int = int(chat_id_str)
            chat = await context.bot.get_chat(chat_id_int)
            await context.bot.copy_message(
                chat_id=chat_id_int,
                from_chat_id=update.effective_chat.id,
                message_id=original_message.message_id
            )
            if chat.type == 'private':
                successful_users += 1
            else:
                successful_groups += 1
                try:
                    count = await context.bot.get_chat_member_count(chat_id_int)
                    total_group_members += count
                except Exception:
                    pass
            await asyncio.sleep(0.1) # Avoid rate limiting
        except Exception as e:
            failed_chats.append(chat_id_str)
            logger.error(f"Error broadcasting message to chat {chat_id_str}: {e}")
    broadcast_end_time = datetime.now()
    duration = broadcast_end_time - broadcast_start_time
    receipt_message = (
        f"**Broadcast Receipt** ✨\n\n"
        f"**Broadcast ID**: `{broadcast_id}`\n"
        f"**Started At**: `{broadcast_start_time.strftime('%Y-%m-%d %H:%M:%S')}`\n"
        f"**Duration**: `{duration.seconds}s`\n\n"
        f"**Summary**\n"
        f"✅ Successful to `{successful_users}` private chats.\n"
        f"✅ Successful to `{successful_groups}` groups.\n"
        f"👥 Total estimated reach (group members + users): `{successful_users + total_group_members}`\n"
        f"❌ Failed for `{len(failed_chats)}` chats.\n\n"
    )
    if failed_chats:
        # Displaying first 10 failed chats
        failed_chats_str = ", ".join(failed_chats[:10]) + ("..." if len(failed_chats) > 10 else "")
        receipt_message += f"**Failed Chat IDs (First 10)**:\n`{failed_chats_str}`"
    await context.bot.send_message(chat_id=chat_id, text=receipt_message, parse_mode='Markdown')
    logger.info(f"Broadcast {broadcast_id} finished. Receipt sent to admin/sudo user.")

# --- NEW: Sudo commands for model switching ---
async def force_gemma_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Forces the bot to use only the Gemma model."""
    global current_model_mode
    current_model_mode = 'gemma_only'
    await update.message.reply_text("🤖 Model mode has been **forced to Gemma-only**.\nAll AI responses will now use Gemma until switched back.", parse_mode='Markdown')
    logger.info(f"Model mode forced to Gemma-only by sudo user {update.effective_user.id}.")

async def force_dynamic_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Switches the bot back to dynamic model selection."""
    global current_model_mode
    current_model_mode = 'dynamic'
    await update.message.reply_text("🤖 Model mode has been reset to **Dynamic**.\nThe bot will now prioritize Gemini and use Gemma/DeepSeek as fallbacks.", parse_mode='Markdown')
    logger.info(f"Model mode set to dynamic by sudo user {update.effective_user.id}.")

# --- NEW: Sudo command for DeepSeek model ---
async def force_deepseek_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Forces the bot to use only the DeepSeek model."""
    global current_model_mode
    current_model_mode = 'deepseek_only'
    await update.message.reply_text("🤖 Model mode has been **forced to DeepSeek-only**.\nAll AI responses will now use DeepSeek until switched back.", parse_mode='Markdown')
    logger.info(f"Model mode forced to DeepSeek-only by sudo user {update.effective_user.id}.")


# --- General User Commands ---
async def get_photo_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message.reply_to_message and update.message.reply_to_message.photo:
        photo_file_id = update.message.reply_to_message.photo[-1].file_id
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Photo File ID:\n`{photo_file_id}`", parse_mode='Markdown')
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Please reply to a photo with this command to get its ID.")

# --- FIXED: Start Command Logic (Used context.bot.get_me()) ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_name = update.effective_user.first_name
    chat_id = update.effective_chat.id
    
    # Save chat ID if new (Sirf session ke liye)
    if str(chat_id) not in known_users:
        known_users.add(str(chat_id))
        # save_chat_id(chat_id) <-- Google Sheets function hata diya
    
    # Fetch bot username for group link
    bot_info = await context.bot.get_me()
    add_to_group_url = f"https://t.me/Siya_aibot?startgroup=true"
    
    keyboard = [[InlineKeyboardButton("➕ Add Me To Your Group ➕", url=add_to_group_url)]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_message = (
        f"Hey there, {user_name}! 😉\n\n"
        f"I'm **Siya**, your friendly AI assistant. 🤖\n" # Siya
        f"I'm here to chat, answer your questions, and help you with anything you need. 😘\n\n"
        f"I also have a new economy and family game! Try it out with `/game`.\n"
        f"You can use `/help` to see all my commands.\n"
        f"Let's get started! 💖"
    )
    photo_file_id = 'AgACAgUAAxkBAAIIKGigVdAK07aRr9QiXpRalahcPO2pAAIL0DEblXUBVSY5LS31XxPSAQADAgADeAADNgQ'
    try:
        await context.bot.send_photo(
            chat_id=chat_id,
            photo=photo_file_id,
            caption=welcome_message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        logger.info(f"[{chat_id}] Sent /start with photo and button to {user_name}")
    except Exception as e:
        # Fallback to text message if photo fails
        logger.error(f"Failed to send photo with start command: {e}")
        await context.bot.send_message(chat_id=chat_id, text=welcome_message, reply_markup=reply_markup, parse_mode='Markdown')

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    ping_start = time.time()
    msg = await context.bot.send_message(chat_id=chat_id, text="Calculating stats...")
    ping_end = time.time()
    
    uptime = datetime.now() - start_time
    hours, remainder = divmod(uptime.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{uptime.days}d {hours}h {minutes}m {seconds}s"
    ram_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent(interval=1)
    disk_usage = psutil.disk_usage('/').percent
    response_text = (
        "❤️ **Siya's Live Stats** ❤️\n\n" # Siya
        f"⚡️ **Ping**: `{int((ping_end - ping_start) * 1000)}ms`\n"
        f"⏳ **Uptime**: `{uptime_str}`\n"
        f"🧠 **RAM**: `{ram_usage}%`\n"
        f"💻 **CPU**: `{cpu_usage}%`\n"
        f"💾 **Disk**: `{disk_usage}%`\n\n"
        "✨ by Nishu ✨"
    )
    
    await msg.delete() # Delete "Calculating stats..." message
    photo_file_id = 'AgACAgUAAxkBAAIIKGigVdAK07aRr9QiXpRalahcPO2pAAIL0DEblXUBVSY5LS31XxPSAQADAgADeAADNgQ'
    try:
        await context.bot.send_photo(
            chat_id=chat_id,
            photo=photo_file_id,
            caption=response_text,
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error(f"Failed to send photo with stats command: {e}")
        await context.bot.send_message(chat_id=chat_id, text=response_text, parse_mode='Markdown')

async def admin_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shows detailed bot stats for the admin/sudo user only."""
    chat_id = update.effective_chat.id
    ping_start = time.time()
    msg = await context.bot.send_message(chat_id=chat_id, text="Calculating admin stats...")
    ping_end = time.time()
    
    uptime = datetime.now() - start_time
    hours, remainder = divmod(uptime.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{uptime.days}d {hours}h {minutes}m {seconds}s"
    ram_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent(interval=1)
    disk_usage = psutil.disk_usage('/').percent
    
    bot_connection_status = "✅ Connected"
    try:
        await context.bot.get_me()
    except Exception:
        bot_connection_status = "❌ Failed"
    
    # --- Sheets connection check hata diya ---
    sheets_connection_status = "⚠️ Disabled (Sheets Removed)"
        
    env_vars_status = "✅ All set"
    if not all([TELEGRAM_BOT_TOKEN, any(GEMINI_API_KEYS), BROADCAST_ADMIN_ID]):
        env_vars_status = "⚠️ Missing key variables"
        
    render_status = "✅ Active" if os.getenv("RENDER") else "⚠️ Local/Unknown"
    
    api_key_status_text = ""
    for i, key in enumerate(GEMINI_API_KEYS):
        if key:
            key_short = key[-5:]
            status = "🟢 Available"
            if time.time() < key_cooldown_until.get(key, 0):
                cooldown_remaining = int(key_cooldown_until[key] - time.time())
                status = f"🟡 Cooldown ({cooldown_remaining}s)"
            api_key_status_text += f"Gemini Key {i+1} (`...{key_short}`): {status}\n"
        else:
            api_key_status_text += f"Gemini Key {i+1}: ❌ Missing\n"
    
    api_key_status_text += f"\nGemma Key: {'✅ Present' if GEMMA_API_KEY else '❌ Missing'}"
    api_key_status_text += f"\nDeepSeek Key: {'✅ Present' if DEEPSEEK_API_KEY else '❌ Missing'}" # NEW
    api_key_status_text += f"\nStable Horde Key: {'✅ Present' if STABLE_HORDE_API_KEY else '❌ Missing'}"
    api_key_status_text += f"\nPexels Key: {'✅ Present' if PEXELS_API_KEY else '❌ Missing'}"
    api_key_status_text += f"\nGenius Key: {'✅ Present' if GENIUS_ACCESS_TOKEN else '❌ Missing'}"
    api_key_status_text += f"\nYouTube Key: {'✅ Present' if YOUTUBE_API_KEY else '⚠️ Missing (Not required by pytube)'}"

    response_text = (
        "👑 **Siya's Admin Report** 👑\n\n" # Siya
        "**System Health**\n"
        f" Ping: `{int((ping_end - ping_start) * 1000)}ms`\n"
        f" Uptime: `{uptime_str}`\n"
        f" RAM: `{ram_usage}%`\n"
        f" CPU: `{cpu_usage}%`\n"
        f" Disk: `{disk_usage}%`\n\n"
        "**Service Status**\n"
        f" Bot Connection: `{bot_connection_status}`\n"
        f" Google Sheets: `{sheets_connection_status}`\n"
        f" Environment Variables: `{env_vars_status}`\n"
        f" Deployment: `{render_status}`\n\n"
        "**Bot Stats**\n"
        f" Total Chats (This Session): `{len(known_users)}`\n" # Updated text
        f" Sudo Users (Hardcoded): `{len(sudo_users)}`\n" # Updated text
        f" Total Messages: `{total_messages_processed}`\n"
        f" AI Model Mode: `{current_model_mode.capitalize()}`\n"
        f" Custom Welcomes: `⚠️ Disabled`\n" # NEW
        "**API Status**\n"
        f"{api_key_status_text}"
        "\n\n✨ by Nishu ✨"
    )
    await msg.delete()
    await context.bot.send_message(chat_id=chat_id, text=response_text, parse_mode='Markdown')
    logger.info(f"[{chat_id}] /adminstats command used by admin/sudo.")

# --- HELPER for show_chats_command (FIXED) ---
async def get_chat_details(chat_id_str: str, bot: Bot):
    """Asynchronously fetches details for a single chat."""
    try:
        chat_id_int = int(chat_id_str)
        chat = await bot.get_chat(chat_id_int)
        
        if chat.type == 'private':
            return f"• **User**: `{chat.full_name}` (ID: `{chat_id_str}`)"
        else:
            members_count = "N/A"
            try:
                members_count = await bot.get_chat_member_count(chat_id_str)
            except Exception: pass
            
            invite_link = ""
            try:
                if chat.username:
                    invite_link = f"https.t.me/{chat.username}"
                else:
                    # Check if bot can create invite link
                    bot_member = await bot.get_chat_member(chat_id_int, bot.id)
                    if bot_member.can_invite_users:
                        invite = await bot.create_chat_invite_link(chat_id_str, member_limit=1)
                        invite_link = invite.invite_link
                    else:
                        invite_link = "*No permission to get link.*"
            except Exception:
                invite_link = "*No permission to get link.*"

            return (
                f"• **Group**: `{chat.title}` (ID: `{chat_id_str}`)\n"
                f"  Members: `{members_count}`\n"
                f"  Link: {invite_link}"
            )
    except Exception as e:
        return f"• **Unknown Chat**: ID: `{chat_id_str}` (Error: Bot was likely removed or chat deleted)"

# --- /show_chats Command (FIXED) ---
async def show_chats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    global known_users
    # --- load_known_users() hata diya ---
    
    known_chats = list(known_users)
    if not known_chats:
        await context.bot.send_message(chat_id=user_id, text="The bot is not in any groups or private chats yet (since last restart).")
        return

    msg = await context.bot.send_message(chat_id=user_id, text=f"Fetching details for `{len(known_chats)}` chats (active this session). This might take a moment...", parse_mode='Markdown')

    # Create async tasks for all chat detail fetches
    tasks = [get_chat_details(chat_id_str, context.bot) for chat_id_str in known_chats]
    # Run them concurrently
    results = await asyncio.gather(*tasks)
    
    chat_details = [r for r in results if r]
    
    response_text = "✨ **Siya's Chats (This Session)** ✨\n\n" + "\n\n".join(chat_details) # Siya
    
    # Send final result in chunks if it's too long
    if len(response_text) > 4096:
        await msg.edit_text("✨ **Siya's Chats** ✨\n\nThe list is too long, sending as multiple messages...") # Siya
        for i in range(0, len(chat_details), 20): # Send 20 chats at a time
            chunk = chat_details[i:i+20]
            chunk_text = "\n\n".join(chunk)
            await context.bot.send_message(user_id, chunk_text, parse_mode='Markdown', disable_web_page_preview=True)
    else:
        await msg.edit_text(response_text, parse_mode='Markdown', disable_web_page_preview=True)
    
    if update.effective_chat.id != user_id:
        await update.message.reply_text("The list of chats has been sent to you privately.")
    
    logger.info(f"[{update.effective_chat.id}] /show_chats command used by sudo user. Response sent privately to {user_id}.")


# --- Sudo-Exclusive Admin Command: Sudo Ban (FIXED & IMPROVED) ---
async def sudo_ban_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Bans a user in a group without being a group admin (requires bot to have admin rights)."""
    chat_id = update.effective_chat.id
    
    target_chat_id = None
    target_user_id = None
    target_user_name = "User"

    if update.message.reply_to_message:
        target_user_id = update.message.reply_to_message.from_user.id
        target_user_name = update.message.reply_to_message.from_user.full_name
        target_chat_id = update.effective_chat.id
    elif context.args and len(context.args) == 2 and context.args[0].startswith("-") and context.args[1].isdigit():
        try:
            target_chat_id = int(context.args[0])
            target_user_id = int(context.args[1])
            target_user_name = f"User ID {target_user_id}"
        except ValueError:
             await update.message.reply_text("Invalid `chat_id` or `user_id` format. Please use correct IDs.")
             return
    else:
        await update.message.reply_text("Please reply to a user's message in a group or use the format: `/s_ban <group_chat_id> <user_id>` (Chat ID starts with a -)", parse_mode='Markdown')
        return

    try:
        # Check if bot can ban
        bot_member = await context.bot.get_chat_member(target_chat_id, context.bot.id)
        if not (bot_member.status in ['administrator', 'creator'] and bot_member.can_restrict_members):
            await update.message.reply_text(f"I don't have the necessary admin rights (Restrict Members) in chat `{target_chat_id}` to ban a user.")
            return

        # Check if target is also an admin
        target_member = await context.bot.get_chat_member(target_chat_id, target_user_id)
        if target_member.status in ['administrator', 'creator']:
            await update.message.reply_text(f"I cannot ban an administrator in chat `{target_chat_id}`.")
            return
            
        await context.bot.ban_chat_member(target_chat_id, target_user_id)
        
        chat_info = await context.bot.get_chat(target_chat_id)
        chat_title = chat_info.title
        
        await context.bot.send_message(
            chat_id=chat_id, 
            text=f"Sudo Ban Successful! ✅\n\n**User**: {target_user_name} (`{target_user_id}`)\n**Banned from Group**: {chat_title} (`{target_chat_id}`)",
            parse_mode='Markdown'
        )
        logger.info(f"[{chat_id}] Sudo user {update.effective_user.id} banned {target_user_id} in chat {target_chat_id}.")
        
    except Exception as e:
        await update.message.reply_text(f"Could not perform sudo ban in chat `{target_chat_id}`. Error: {e}")
        logger.error(f"[{chat_id}] Error in sudo ban for user {target_user_id} in chat {target_chat_id}: {e}")

# --- NEW Sudo Command: Sudo Unban ---
async def sudo_unban_user(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Unbans a user in a group using the bot's rights."""
    chat_id = update.effective_chat.id
    
    # 1. Argument Parsing
    target_chat_id = None
    target_user_id = None
    target_user_name = "User"

    if update.message.reply_to_message:
        target_user_id = update.message.reply_to_message.from_user.id
        target_user_name = update.message.reply_to_message.from_user.full_name
        target_chat_id = update.effective_chat.id
    elif context.args and len(context.args) == 2 and context.args[0].startswith("-") and context.args[1].isdigit():
        # Usage: /s_unban <target_chat_id> <target_user_id>
        try:
            target_chat_id = int(context.args[0])
            target_user_id = int(context.args[1])
            target_user_name = f"User ID {target_user_id}"
        except ValueError:
             await update.message.reply_text("Invalid `chat_id` or `user_id` format. Please use correct IDs.")
             return
    else:
        await update.message.reply_text("Please reply to a user's message in a group or use the format: `/s_unban <group_chat_id> <user_id>` (Chat ID starts with a -)", parse_mode='Markdown')
        return

    try:
        # 2. Check if bot can unban (it must be an admin with restrict members rights)
        bot_member = await context.bot.get_chat_member(target_chat_id, context.bot.id)
        if not (bot_member.status in ['administrator', 'creator'] and bot_member.can_restrict_members):
            await update.message.reply_text(f"I don't have the necessary admin rights (Restrict Members) in chat `{target_chat_id}` to unban a user.")
            return

        # 3. Perform the unban (only_if_banned=True is the default for unban)
        await context.bot.unban_chat_member(target_chat_id, target_user_id)
        
        # 4. Success Message
        chat_title = (await context.bot.get_chat(target_chat_id)).title
        await context.bot.send_message(
            chat_id=chat_id, 
            text=f"Sudo Unban Successful! ✅\n\n**User**: {target_user_name} (`{target_user_id}`)\n**Unbanned from Group**: {chat_title} (`{target_chat_id}`)",
            parse_mode='Markdown'
        )
        logger.info(f"[{chat_id}] Sudo user {update.effective_user.id} unbanned {target_user_id} in chat {target_chat_id}.")
        
    except Exception as e:
        await update.message.reply_text(f"Could not perform sudo unban in chat `{target_chat_id}`. Error: {e}")
        logger.error(f"[{chat_id}] Error in sudo unban for user {target_user_id} in chat {target_chat_id}: {e}")

# --- NEW Sudo Command: Sudo Send Message ---
async def sudo_send_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a message to a specific chat ID from the bot."""
    chat_id = update.effective_chat.id
    
    if len(context.args) < 2:
        await update.message.reply_text("Please provide a target chat ID and a message. Usage: `/s_send <chat_id> <message>`", parse_mode='Markdown')
        return

    target_chat_id_str = context.args[0]
    message_text = " ".join(context.args[1:])

    try:
        target_chat_id = int(target_chat_id_str)
    except ValueError:
        await update.message.reply_text("Invalid chat ID provided.")
        return

    try:
        # Send the message
        await context.bot.send_message(chat_id=target_chat_id, text=message_text, parse_mode='Markdown')
        
        await update.message.reply_text(f"Message successfully sent to chat `{target_chat_id}`.")
        logger.info(f"[{chat_id}] Sudo user {update.effective_user.id} sent message to chat {target_chat_id}: {message_text[:50]}...")
    except Exception as e:
        await update.message.reply_text(f"Could not send message to chat `{target_chat_id}`: {e}")
        logger.error(f"[{chat_id}] Error in sudo send message to chat {target_chat_id}: {e}")


# --- ALL OLD MONEY GAME COMMANDS (account, farm, work, etc.) ARE DELETED ---
# --- They are now handled by game.py ---


# --- NEW: Fun Command Data (EXPANDED) ---
TRUTH_QUESTIONS = [
    "What's the most embarrassing thing you've ever done?", "What's a secret you've never told anyone?",
    "Who is your secret crush?", "What's the biggest lie you've ever told?", "What's your biggest fear?",
    "What's something you're self-conscious about?", "Have you ever cheated on a test?",
    "What's the most childish thing you still do?", "What's a weird food combination you love?",
    "Have you ever peed in a swimming pool?", "What's the cringiest text message you've ever sent?",
    "Who do you stalk the most on social media?", "What's the silliest reason you've ever cried?",
    "If you could trade lives with someone for a day, who would it be and why?", "What's your guilty pleasure song?",
    "Have you ever lied to get out of a date?", "What's the most trouble you've ever been in?",
    "What's one thing you would change about your appearance?", "What's a secret talent you have?",
    "What's the worst advice you've ever given someone?", "Have you ever pretended to be sick to skip school or work?",
    "What's the most embarrassing nickname you've ever had?",
]

DARES = [
    "Send a screenshot of your home screen.", "Talk in a funny accent for the next 5 minutes.",
    "Send the 5th picture from your gallery.", "Post 'I love Siya Bot' as your status.", # Siya
    "Send a voice message singing a song.", "Let the group choose a new profile picture for you for the next 24 hours.",
    "Try to lick your elbow.", "Spell your name backwards with your nose.", "Send an old, embarrassing photo of yourself.",
    "Text your crush and say 'I know your secret'.", "Do 10 push-ups right now and send a video proof.",
    "Eat a spoonful of a weird condiment (like ketchup or mustard).", "DM a celebrity on Instagram.",
    "Change your name in the group to 'Siya's Biggest Fan' for an hour.", # Siya
    "Try to juggle three random objects.", "Speak only in emojis for the next 10 minutes.",
    "Send a voice note imitating your favorite cartoon character.",
    "Write a short, silly poem about the person who sent the last message.", "Go outside and shout 'I believe in unicorns!'.",
    "Stack as many books as you can on your head and take a picture.", "Draw a self-portrait with your eyes closed and share it.",
    "Find the weirdest item in your room and explain its story.",
]

SLAP_MESSAGES = [
    "slaps {target} around a bit with a large trout! 🐟",
    "gives {target} a high-five. In the face. With a chair. 🪑",
    "summons a mighty slap for {target}! 💥",
    "throws a wet noodle at {target}. It's surprisingly effective. 🍝",
    "tickles {target} until they promise to behave. ✨",
]

ROAST_MESSAGES = [
    "I'd agree with you but then we’d both be wrong.",
    "You have your entire life to be a jerk. Why not take today off?",
    "I'm not insulting you, I'm describing you.",
    "Is your brain made of sponges? Because it seems to soak everything up but does nothing with it.",
    "You're the reason the gene pool needs a lifeguard.",
    "If I wanted to hear from an idiot, I would've watched the news.",
    "I've been called worse things by better people.",
]

# --- NEW: Fun & Game Commands ---
async def truth_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_name = update.effective_user.first_name
    await update.message.reply_text(f"Okay, {user_name}! Your truth is:\n\n**{random.choice(TRUTH_QUESTIONS)}**")

async def dare_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_name = update.effective_user.first_name
    await update.message.reply_text(f"Here you go, {user_name}! I dare you to:\n\n**{random.choice(DARES)}**")

async def slap_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    slapper = update.effective_user.first_name
    
    if update.message.reply_to_message:
        target = update.message.reply_to_message.from_user.first_name
        message = random.choice(SLAP_MESSAGES).format(target=target)
        await update.message.reply_text(f"{slapper} {message}")
    else:
        await update.message.reply_text(f"{slapper} slaps themselves in confusion! 🤔")

async def coinflip_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    result = random.choice(["Heads", "Tails"])
    await update.message.reply_text(f"The coin landed on... **{result}**! 🪙")

async def roll_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    result = random.randint(1, 6)
    await update.message.reply_text(f"You rolled a... **{result}**! 🎲")

async def start_game_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Starts a number guessing game in the chat."""
    chat_id = update.effective_chat.id
    if guessing_games[chat_id].get('active', False):
        await update.message.reply_text("A game is already in progress in this chat! Use /stopgame to end it.")
        return
    
    secret_number = random.randint(1, 100)
    guessing_games[chat_id] = {'number': secret_number, 'attempts': 0, 'active': True}
    
    logger.info(f"[{chat_id}] New guessing game started. Number is {secret_number}.")
    await update.message.reply_text("🎲 **Number Guessing Game Started!** 🎲\n\nI'm thinking of a number between 1 and 100. Try to guess it by sending a number in the chat!")

async def stop_game_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Stops the current number guessing game."""
    chat_id = update.effective_chat.id
    if not guessing_games[chat_id].get('active', False):
        await update.message.reply_text("There's no game in progress to stop.")
        return
        
    secret_number = guessing_games[chat_id].get('number')
    guessing_games[chat_id].clear()
    
    await update.message.reply_text(f"Game stopped! The number was {secret_number}. Use /startgame to play again.")
    logger.info(f"[{chat_id}] Game stopped by user.")


# ---  NEW UTILITY AND FUN COMMANDS (15+) ---

# Utility Commands
async def weather_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gets the weather for a specified city using OpenWeatherMap or Visual Crossing as a fallback."""
    if not context.args:
        await update.message.reply_text("Please provide a city name. Example: `/weather London`")
        return

    city = " ".join(context.args)
    
    # Try OpenWeatherMap first
    if WEATHER_API_KEY:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        try:
            response = requests.get(url).json()
            if response["cod"] == 200:
                main = response["main"]
                wind = response["wind"]
                weather_desc = response["weather"][0]["description"]
                weather_icon = response["weather"][0]["icon"]
                
                icon_map = {
                    "01": "☀️", "02": "🌤️", "03": "☁️", "04": "☁️", "09": "🌧️",
                    "10": "🌦️", "11": "⛈️", "13": "❄️", "50": "🌫️"
                }
                emoji = icon_map.get(weather_icon[:2], "🌡️")
                
                weather_text = (
                    f"**Weather in {response['name']}, {response['sys']['country']}** {emoji}\n\n"
                    f"🌡️ **Temperature**: {main['temp']}°C (Feels like: {main['feels_like']}°C)\n"
                    f"📊 **Pressure**: {main['pressure']} hPa\n"
                    f"💧 **Humidity**: {main['humidity']}%\n"
                    f"🌬️ **Wind**: {wind['speed']} m/s\n"
                    f"📜 **Condition**: {weather_desc.title()}"
                )
                await update.message.reply_text(weather_text, parse_mode='Markdown')
                return
        except Exception as e:
            logger.error(f"Error in /weather command with OpenWeatherMap: {e}")

    # Fallback to Visual Crossing
    if VISUALCROSSING_API_KEY:
        url = f"https.weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}?unitGroup=metric&key={VISUALCROSSING_API_KEY}&contentType=json"
        try:
            response = requests.get(url).json()
            if "address" in response:
                current = response["currentConditions"]
                
                icon_map = {
                    "clear-day": "☀️", "clear-night": "🌙", "partly-cloudy-day": "🌤️",
                    "partly-cloudy-night": "☁️", "cloudy": "☁️", "rain": "🌧️",
                    "showers-day": "🌦️", "showers-night": "🌦️", "snow": "❄️", "thunder-rain": "⛈️",
                    "fog": "🌫️"
                }
                emoji = icon_map.get(current.get("icon"), "🌡️")

                weather_text = (
                    f"**Weather in {response['resolvedAddress']}** {emoji}\n\n"
                    f"🌡️ **Temperature**: {current.get('temp', 'N/A')}°C (Feels like: {current.get('feelslike', 'N/A')}°C)\n"
                    f"💧 **Humidity**: {current.get('humidity', 'N/A')}%\n"
                    f"🌬️ **Wind**: {current.get('windspeed', 'N/A')} km/h\n"
                    f"📜 **Condition**: {current.get('conditions', 'N/A')}"
                )
                await update.message.reply_text(weather_text, parse_mode='Markdown')
                return
        except Exception as e:
            await update.message.reply_text("An error occurred while fetching weather data.")
            logger.error(f"Error in /weather command with Visual Crossing: {e}")
            return

    await update.message.reply_text("Sorry, both weather API services are currently unavailable or keys are missing.")


async def crypto_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gets the price of a cryptocurrency."""
    if not context.args:
        # --- FIX: Changed example from 'btc' to 'bitcoin' ---
        await update.message.reply_text("Please provide a crypto symbol. Example: `/crypto bitcoin`")
        return
        
    symbol = context.args[0].lower()
    url = f"https.api.coingecko.com/api/v3/simple/price?ids={symbol}&vs_currencies=usd,inr&include_24hr_change=true"
    
    try:
        response = requests.get(url).json()
        if symbol in response:
            data = response[symbol]
            usd_price = data.get('usd', 'N/A')
            inr_price = data.get('inr', 'N/A')
            change_24h = data.get('usd_24h_change', 0)
            emoji = "📈" if change_24h >= 0 else "📉"
            
            crypto_text = (
                f"**Crypto Price for {symbol.upper()}**\n\n"
                f"🇺🇸 **USD**: `${usd_price:,.2f}`\n"
                f"🇮🇳 **INR**: `₹{inr_price:,.2f}`\n"
                f"{emoji} **24h Change**: `{change_24h:.2f}%`"
            )
            await update.message.reply_text(crypto_text, parse_mode='Markdown')
        else:
            await update.message.reply_text(f"Could not find price data for `{symbol}`. Make sure to use the coin's ID (e.g., 'bitcoin', 'ethereum').")
    except Exception as e:
        await update.message.reply_text("An error occurred while fetching crypto data.")
        logger.error(f"Error in /crypto command: {e}")
        
async def define_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gets the definition of a word."""
    if not context.args:
        await update.message.reply_text("Please provide a word to define. Example: `/define integrity`")
        return
    
    word = " ".join(context.args)
    url = f"https.api.dictionaryapi.dev/api/v2/entries/en/{word}"
    
    try:
        response = requests.get(url).json()
        if isinstance(response, list) and "meanings" in response[0]:
            meanings = response[0]['meanings']
            definition_text = f"**Definitions for: {word.title()}**\n\n"
            for meaning in meanings[:3]: # Show first 3 parts of speech
                part_of_speech = meaning.get('partOfSpeech', '')
                definition_text += f"_{part_of_speech.title()}_\n"
                for i, definition in enumerate(meaning['definitions'][:2]): # Show first 2 definitions
                    definition_text += f"{i+1}. {definition['definition']}\n"
                    if 'example' in definition:
                        definition_text += f"   *Example: \"{definition['example']}\"*\n"
                definition_text += "\n"
            await update.message.reply_text(definition_text, parse_mode='Markdown')
        else:
            await update.message.reply_text(f"Sorry, I couldn't find a definition for `{word}`.")
    except Exception as e:
        await update.message.reply_text("An error occurred while fetching the definition.")
        logger.error(f"Error in /define command: {e}")

async def qr_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generates a QR code from text."""
    if not context.args:
        await update.message.reply_text("Please provide text or a link to generate a QR code. Example: `/qr https://google.com`")
        return
        
    text = " ".join(context.args)
    encoded_text = urllib.parse.quote(text)
    qr_url = f"https.api.qrserver.com/v1/create-qr-code/?size=500x500&data={encoded_text}"
    
    await context.bot.send_photo(
        chat_id=update.effective_chat.id,
        photo=qr_url,
        caption=f"Here is the QR code for: `{text}`",
        parse_mode='Markdown'
    )

async def shorten_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Shortens a URL using TinyURL."""
    if not TINYURL_API_KEY:
        await update.message.reply_text("Sorry, the URL shortener API key is not configured.")
        return
    if not context.args:
        await update.message.reply_text("Please provide a URL to shorten. Example: `/shorten <your_long_url>`")
        return
        
    url = context.args[0]
    api_url = f"https.api.tinyurl.com/create?api_token={TINYURL_API_KEY}"
    payload = {"url": url, "domain": "tiny.one"} # Using tiny.one domain
    
    try:
        response = requests.post(api_url, json=payload)
        if response.status_code == 200 and response.json()["code"] == 0:
            short_url = response.json()["data"]["tiny_url"]
            await update.message.reply_text(f"🔗 **Shortened URL**:\n{short_url}")
        else:
            error = response.json().get("errors", ["Unknown error"])[0]
            await update.message.reply_text(f"Could not shorten URL. Error: {error}")
    except Exception as e:
        await update.message.reply_text("An error occurred while shortening the URL.")
        logger.error(f"Error in /shorten command: {e}")

async def ud_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Searches Urban Dictionary."""
    if not context.args:
        await update.message.reply_text("Please provide a term to search on Urban Dictionary. Example: `/ud bruh`")
        return
    term = " ".join(context.args)
    url = f"http://api.urbandictionary.com/v0/define?term={term}"
    try:
        response = requests.get(url).json()
        if response and response.get('list'):
            top_def = response['list'][0]
            definition = re.sub(r'[\[\]]', '', top_def['definition'])
            example = re.sub(r'[\[\]]', '', top_def['example'])
            result_text = (
                f"**{top_def['word']}** on Urban Dictionary\n\n"
                f"**Definition**:\n{definition}\n\n"
                f"**Example**:\n_{example}_\n\n"
                f"👍 {top_def['thumbs_up']} | 👎 {top_def['thumbs_down']}"
            )
            await update.message.reply_text(result_text)
        else:
            await update.message.reply_text(f"Couldn't find a definition for `{term}` on Urban Dictionary.")
    except Exception as e:
        await update.message.reply_text("An error occurred while searching Urban Dictionary.")
        logger.error(f"Error in /ud command: {e}")

# --- FIXED: /paste command ---
async def paste_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Pastes replied text to a pastebin service."""
    if not update.message.reply_to_message or not update.message.reply_to_message.text:
        await update.message.reply_text("Please reply to a text message to paste it.")
        return
    
    text_to_paste = update.message.reply_to_message.text
    url = "https.spaceb.in/api/v1/documents" # This is a POST endpoint, not for direct data sending
    
    try:
        # The correct way is to send the text as the 'data' payload
        response = requests.post(url, data=text_to_paste.encode('utf-8'))
        response.
