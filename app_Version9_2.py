import base64
import logging
import os
import time
import uvicorn
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import httpx
import requests
import sqlite3
from authlib.integrations.starlette_client import OAuth
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from PIL import Image
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from loguru import logger

# Configure logging with Loguru
logger.remove()  # Remove default handler
logger.add(
    "app.log",
    rotation="10 MB",
    compression="zip",
    level="INFO",
    format="{time} {level} {message}",
    enqueue=True
)
logger.add(
    os.sys.stderr,
    level="INFO",
    format="{time} {level} {message}"
)

# Environment Variables
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://front-end-bpup.vercel.app")
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "changeme_secret_key_123456")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
STABILITY_API_URL = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
VIRTUSIM_API_KEY = os.getenv("VIRTUSIM_API_KEY")
VIRTUSIM_API_URL = "https://virtusim.com/api/"

# Constants
ADMIN_USERS = ["admin@kugy.ai", "testadmin"]
ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:3000",
    "https://front-end-bpup.vercel.app",
]

# Initialize FastAPI
app = FastAPI()

# Configure middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY, max_age=3600)

def init_db():
    """Initialize SQLite database with required tables."""
    with sqlite3.connect("credits.db") as conn:
        c = conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                user_name TEXT,
                credits INTEGER,
                login_streak INTEGER,
                last_login TEXT,
                last_guest_timestamp INTEGER,
                last_reward_date TEXT
            )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                question TEXT,
                answer TEXT,
                created_at TEXT
            )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS virtual_numbers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                phone_number TEXT,
                provider TEXT,
                purchase_date TEXT,
                status TEXT,
                price INTEGER,
                service_id TEXT,
                country TEXT
            )"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS number_purchases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                phone_number TEXT,
                provider TEXT,
                price INTEGER,
                purchase_date TEXT,
                status TEXT
            )"""
        )
        conn.commit()

init_db()

def check_credits(user_id: str, need: int = 1) -> bool:
    """Check if user has enough credits and deduct if available."""
    if not user_id or user_id in ADMIN_USERS:
        return True

    with sqlite3.connect("credits.db") as conn:
        c = conn.cursor()
        c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
        result = c.fetchone()
        if not result or result[0] < need:
            logger.warning(f"User {user_id} does not have enough credits. Has {result[0] if result else 0}, needs {need}.")
            return False
        c.execute("UPDATE users SET credits = credits - ? WHERE user_id = ?", (need, user_id))
        conn.commit()
        logger.info(f"User {user_id} credits reduced by {need}. Remaining: {result[0] - need}.")
        return True

def get_credits(user_id: str) -> str:
    """Get user's credit balance."""
    if not user_id:
        return "0"
    if user_id in ADMIN_USERS:
        return "∞"
    with sqlite3.connect("credits.db") as conn:
        c = conn.cursor()
        c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
        result = c.fetchone()
        return str(result[0]) if result else "0"

def add_or_init_user(user_id: str, user_name: str = "User"):
    """Add or initialize user in database."""
    default_credits = 75 if "@" in user_id else 25
    if user_name == "Guest":
        default_credits = 25

    with sqlite3.connect("credits.db") as conn:
        c = conn.cursor()
        c.execute(
            """INSERT OR IGNORE INTO users 
            (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date) 
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                user_name,
                default_credits,
                0,
                "2025-06-10 22:14:51",
                int(time.time()),
                "",
            ),
        )
        conn.commit()

def save_chat_history(user_id: str, question: str, answer: str):
    """Save chat history to database."""
    with sqlite3.connect("credits.db") as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO chat_history (user_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
            (user_id, question, answer, "2025-06-10 22:14:51"),
        )
        conn.commit()
        logger.info(f"Chat history saved for user {user_id}")

def get_chat_history(user_id: str, limit: int = 20) -> List[Dict[str, str]]:
    """Retrieve chat history for a user."""
    with sqlite3.connect("credits.db") as conn:
        c = conn.cursor()
        c.execute(
            "SELECT question, answer, created_at FROM chat_history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )
        history = [
            {"question": row[0], "answer": row[1], "created_at": row[2]} for row in c.fetchall()
        ][::-1]
        logger.info(f"Retrieved {len(history)} chat entries for user {user_id}")
        return history

# Configure OAuth
oauth = OAuth()
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile", "state": lambda: os.urandom(16).hex()},
)

# VirtuSim Service
class VirtuSimService:
    def __init__(self):
        self.api_key = VIRTUSIM_API_KEY
        self.base_url = VIRTUSIM_API_URL
        self.service_user = "lillysummer9794"

    async def get_available_services(self, country: str = "indonesia", service: str = "wa") -> Dict[str, Any]:
        """Get real-time service availability and pricing."""
        if not self.api_key:
            logger.error("VIRTUSIM_API_KEY is not set")
            return {
                "status": "error",
                "message": "VirtuSim API key missing",
                "timestamp": "2025-06-10 22:14:51",
                "user": self.service_user
            }

        async with httpx.AsyncClient() as client:
            try:
                url = f"{self.base_url}"
                params = {
                    "api_key": self.api_key,
                    "action": "services",
                    "country": country.lower() if country else "",
                    "service": service.lower() if service else ""
                }

                logger.info(f"VIRTUSIM API Request - URL: {url}")
                logger.info(f"Params: {params}")
                
                response = await client.get(
                    url,
                    params=params,
                    timeout=30.0
                )
                
                logger.info(f"Status Code: {response.status_code}")
                logger.info(f"Raw Response: {response.text[:500]}...")

                if response.status_code == 200:
                    try:
                        data = response.json()
                        services = []
                        raw_services_list = []

                        if isinstance(data, dict):
                            if "data" in data:
                                raw_services_list = data["data"]
                            elif "error" in data:
                                return {
                                    "status": "error",
                                    "message": f"VirtuSim API Error: {data.get('error')}",
                                    "timestamp": "2025-06-10 22:14:51",
                                    "user": self.service_user
                                }
                        elif isinstance(data, list):
                            raw_services_list = data
                        
                        for service_item in raw_services_list:
                            try:
                                stock = int(service_item.get("stock", 0))
                                if stock > 0:
                                    price = float(service_item.get("price", 0))
                                    services.append({
                                        "service_id": str(service_item.get("id", "")),
                                        "name": service_item.get("name", "Unknown"),
                                        "description": f"Verifikasi {service_item.get('name')} dengan nomor virtual",
                                        "price": price,
                                        "price_formatted": f"Rp {price:,.2f}",
                                        "available_numbers": stock,
                                        "country": service_item.get("country", country).capitalize(),
                                        "status": "available",
                                        "duration": service_item.get("validity", "48 jam"),
                                        "is_promo": bool(service_item.get("is_promo", False)),
                                        "category": service_item.get("category", "OTP")
                                    })
                            except ValueError as e:
                                logger.error(f"Error processing service: {e}")
                                continue

                        logger.info(f"Successfully processed {len(services)} available services")
                        return {
                            "status": "success",
                            "data": services,
                            "contact": {
                                "whatsapp": "wa.me/+628xxxxxxxx",
                                "discord": "discord.gg/xxxxx"
                            } if services else None,
                            "timestamp": "2025-06-10 22:14:51",
                            "user": self.service_user
                        }
                    except Exception as e:
                        logger.error(f"Error parsing response: {e}")
                        raise

                logger.error(f"API Error: {response.status_code} - {response.text}")
                return {
                    "status": "error", 
                    "message": f"API Error: {response.status_code}",
                    "timestamp": "2025-06-10 22:14:51",
                    "user": self.service_user
                }

            except Exception as e:
                logger.error(f"Request Error: {e}")
                return {
                    "status": "error",
                    "message": str(e),
                    "timestamp": "2025-06-10 22:14:51",
                    "user": self.service_user
                }

    async def get_supported_countries(self) -> List[str]:
        """Return list of supported countries."""
        logger.info("Providing hardcoded list of supported countries for VirtuSim")
        return [
            "Indonesia", "Russia", "Vietnam", "Kazakhstan", "Ukraine",
            "Philippines", "Thailand", "Malaysia", "India", "China"
        ]

# Initialize VirtuSim service
virtusim_service = VirtuSimService()

# Pydantic Models
class ChatRequest(BaseModel):
    user_email: str
    message: str
    model_select: str = "x-ai/grok-3-mini-beta"

class ImageRequest(BaseModel):
    user_email: str
    prompt: str

class VirtuSimPurchase(BaseModel):
    country: str
    service_id: str
    user_email: str

# API Endpoints
@app.get("/auth/google")
async def login_via_google(request: Request):
    """Initiate Google OAuth login."""
    redirect_uri = request.url_for("auth_google_callback")
    state = os.urandom(16).hex()
    request.session["oauth_state"] = state
    logger.info(f"Initiating Google OAuth login. Redirect URI: {redirect_uri}")
    return await oauth.google.authorize_redirect(request, redirect_uri, state=state)

@app.get("/auth/google/callback")
async def auth_google_callback(request: Request):
    """Handle Google OAuth callback."""
    try:
        expected_state = request.session.get("oauth_state")
        if not expected_state:
            logger.error("No state in session")
            return RedirectResponse(url=f"{FRONTEND_URL}/?error=no_state")

        token = await oauth.google.authorize_access_token(request)
        actual_state = request.query_params.get("state")
        if actual_state != expected_state:
            logger.error(f"Mismatching state: Expected {expected_state}, got {actual_state}")
            return RedirectResponse(url=f"{FRONTEND_URL}/?error=csrf_warning")

        del request.session["oauth_state"]

        user = token.get("userinfo") or await oauth.google.parse_id_token(request, token)
        if not user:
            resp = await oauth.google.get("userinfo", token=token)
            user = resp.json()

        email = user.get("email", "")
        if not email:
            logger.error("Google OAuth: Email not found in user profile")
            return RedirectResponse(url=f"{FRONTEND_URL}/?error=oauth_no_email")

        add_or_init_user(email, user.get("name", "User"))
        logger.info(f"Google OAuth successful for user: {email}")
        return RedirectResponse(url=f"{FRONTEND_URL}/menu?email={email}")
    except Exception as e:
        logger.error(f"Google OAuth callback error: {e}")
        return RedirectResponse(url=f"{FRONTEND_URL}/?error=oauth_failed")

@app.post("/api/chat")
async def ai_chat(req: ChatRequest):
    """Handle AI chat requests."""
    if not req.user_email:
        raise HTTPException(status_code=401, detail="UNAUTHORIZED")

    add_or_init_user(req.user_email, req.user_email)
    if not check_credits(req.user_email, 1):
        raise HTTPException(status_code=402, detail="NOT_ENOUGH_CREDITS")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://kugy.ai",
        "X-Title": "KugyAI",
    }
    model_map = {
        "OpenRouter (Grok 3 Mini Beta)": "x-ai/grok-3-mini-beta",
        "OpenRouter (Gemini 2.0 Flash)": "google/gemini-flash-1.5",
    }
    payload = {
        "model": model_map.get(req.model_select, "x-ai/grok-3-mini-beta"),
        "messages": [
            {"role": "system", "content": "Act as a helpful assistant named Kugy.ai."},
            {"role": "user", "content": req.message},
        ],
        "temperature": 0.7,
    }

    try:
        logger.info(f"Sending chat request to OpenRouter for user {req.user_email}")
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]
        save_chat_history(req.user_email, req.message, reply)
        return {"reply": reply, "credits": get_credits(req.user_email)}
    except Exception as e:
        logger.error(f"OpenRouter error: {e}")
        raise HTTPException(status_code=503, detail="AI Service Unavailable")

@app.post("/api/generate-image")
async def generate_image(req: ImageRequest):
    """Generate image using Stability AI."""
    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    }

    if not req.user_email:
        return JSONResponse({"error": "UNAUTHORIZED"}, status_code=401, headers=cors_headers)
    if not check_credits(req.user_email, 10):
        return JSONResponse(
            {"error": "NOT_ENOUGH_CREDITS", "message": "Need 10 credits"},
            status_code=402,
            headers=cors_headers,
        )
    if not STABILITY_API_KEY:
        return JSONResponse(
            {"error": "API_KEY_MISSING", "message": "Stability AI API key not set"},
            status_code=503,
            headers=cors_headers,
        )

    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "text_prompts": [{"text": req.prompt}],
        "cfg_scale": 7,
        "height": 1024,
        "width": 1024,
        "samples": 1,
        "steps": 30,
    }

    try:
        logger.info(f"Sending image generation request for user {req.user_email}")
        response = requests.post(STABILITY_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        if "artifacts" in data and data["artifacts"]:
            return JSONResponse(
                {
                    "image": data["artifacts"][0]["base64"],
                    "credits": get_credits(req.user_email),
                    "message": "Kugy.ai: Ini gambarnya buat ayang, cute banget kan? 🐱",
                },
                headers=cors_headers,
            )
        return JSONResponse(
            {"error": "Failed to get image from Stability AI"}, status_code=500, headers=cors_headers
        )
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return JSONResponse(
            {"error": f"Error generating image: {str(e)}"}, status_code=500, headers=cors_headers
        )

@app.get("/api/virtusim/services")
async def get_virtusim_services(country: str = Query("Indonesia"), service: str = Query("Whatsapp")):
    """Get VirtuSim services."""
    try:
        logger.info("=== VIRTUSIM SERVICE REQUEST STARTED ===")
        logger.info(f"Timestamp: 2025-06-10 22:14:51")
        logger.info(f"User: lillysummer9794")
        logger.info(f"Country: {country}")
        logger.info(f"Service: {service}")
        
        result = await virtusim_service.get_available_services(country, service)
        logger.info("=== VIRTUSIM SERVICE REQUEST COMPLETED ===")
        
        return JSONResponse({
            "status": "success",
            "data": result.get("data", []),
            "contact": result.get("contact"),
            "timestamp": "2025-06-10 22:14:51",
            "user": "lillysummer9794"
        })
    except Exception as e:
        logger.error(f"=== VIRTUSIM SERVICE REQUEST ERROR ===")
        logger.error(f"Error: {str(e)}")
        logger.error(f"Timestamp: 2025-06-10 22:14:51")
        logger.error(f"User: lillysummer9794")
        return JSONResponse(
            {
                "status": "error",
                "message": str(e),
                "timestamp": "2025-06-10 22:14:51",
                "user": "lillysummer9794"
            },
            status_code=500,
        )

@app.get("/api/virtusim/countries")
async def get_virtusim_countries():
    """Get supported VirtuSim countries."""
    try:
        countries = await virtusim_service.get_supported_countries()
        return JSONResponse(
            {
                "status": "success",
                "data": countries,
                "timestamp": "2025-06-10 22:14:51",
                "user": "lillysummer9794"
            }
        )
    except Exception as e:
        logger.error(f"Error getting countries: {e}")
        return JSONResponse(
            {
                "status": "error",
                "message": str(e),
                "timestamp": "2025-06-10 22:14:51",
                "user": "lillysummer9794"
            },
            status_code=500,
        )

@app.post("/api/virtusim/purchase")
async def purchase_virtusim_service(purchase: VirtuSimPurchase):
    """Handle VirtuSim service purchase."""
    try:
        service_info = await virtusim_service.get_available_services(purchase.country, purchase.service_id)
        if not service_info.get("data"):
            return JSONResponse(
                {
                    "status": "error",
                    "message": "Service not available",
                    "timestamp": "2025-06-10 22:14:51",
                    "user": purchase.user_email,
                },
                status_code=400,
            )

        service_details = service_info["data"][0]
        return JSONResponse(
            {
                "status": "success",
                "message": "Silakan hubungi admin untuk melanjutkan pembelian",
                "service_details": {
                    "name": service_details["name"],
                    "price": service_details["price_formatted"],
                    "duration": service_details["duration"],
                    "country": purchase.country,
                    "available": service_details["available_numbers"],
                    "status": service_details["status"],
                },
                "contact": {"whatsapp": "wa.me/+628xxxxxxxx", "discord": "discord.gg/xxxxx"},
                "timestamp": "2025-06-10 22:14:51",
                "user": purchase.user_email,
            }
        )
    except Exception as e:
        logger.error(f"Error in purchase request: {e}")
        return JSONResponse(
            {
                "status": "error",
                "message": str(e),
                "timestamp": "2025-06-10 22:14:51",
                "user": purchase.user_email,
            },
            status_code=500,
        )

@app.get("/api/credits")
async def api_credits(user_email: str):
    """Get user credits."""
    return {"credits": get_credits(user_email)}

@app.get("/api/history")
async def api_history(user_email: str = Query(...), limit: int = Query(20, le=100)):
    """Get chat history."""
    if not user_email:
        raise HTTPException(status_code=401, detail="UNAUTHORIZED")
    return {"history": get_chat_history(user_email, limit)}

@app.post("/api/guest-login")
async def guest_login(request: Request):
    """Handle guest login."""
    data = await request.json()
    user_email = data.get("email")
    if not user_email:
        raise HTTPException(status_code=400, detail="Email wajib diisi")

    add_or_init_user(user_email, "Guest")
    dummy_token = f"guest-token-{user_email.split('@')[0]}"
    return JSONResponse(
        {
            "token": dummy_token,
            "credits": get_credits(user_email),
            "message": "Kugy.ai: Mode tamu aktif! 😺",
        }
    )

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "status": "ok",
        "message": "API is running",
        "timestamp": "2025-06-10 22:14:51",
        "user": "lillysummer9794"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)