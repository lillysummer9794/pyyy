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

# Environment Variables
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://front-end-bpup.vercel.app")
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "changeme_secret_key_123456")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")
VIRTUSIM_API_KEY = os.getenv("VIRTUSIM_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
STABILITY_API_URL = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"

# Constants
ADMIN_USERS = ["admin@kugy.ai", "testadmin"]
ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:3000",
    "https://front-end-bpup.vercel.app",
]
CALLBACK_URL = "https://backend-cb98.onrender.com/auth/google/callback"
DB_PATH = os.getenv("DB_PATH", "credits.db")
GUEST_INITIAL_CREDITS = 25  # Fixed guest credits to 25

def ensure_db_and_log():
    """Ensure database and log file exist with proper initialization"""
    try:
        # Configure logging
        logger.remove()
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
        
        # Initialize database with admin users
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            # Create users table
            c.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    user_name TEXT,
                    credits INTEGER,
                    login_streak INTEGER,
                    last_login TEXT,
                    last_guest_timestamp INTEGER,
                    last_reward_date TEXT
                )
            ''')
            
            # Create chat_history table
            c.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    question TEXT,
                    answer TEXT,
                    created_at TEXT
                )
            ''')
            
            # Insert admin users if they don't exist
            admin_users = [
                ("admin@kugy.ai", "Admin", 999999, 0, "2025-06-11 18:12:33", 0, ""),
                ("testadmin", "Test Admin", 999999, 0, "2025-06-11 18:12:33", 0, "")
            ]
            
            c.executemany('''
                INSERT OR IGNORE INTO users 
                (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', admin_users)
            
            conn.commit()
            
        logger.info("Database and log file initialized successfully")
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise

# Initialize FastAPI
app = FastAPI()

# Configure middleware with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY, max_age=3600)

def check_credits(user_id: str, need: int = 1) -> bool:
    """Check if user has enough credits and deduct if available."""
    if not user_id or user_id in ADMIN_USERS:
        return True

    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
            result = c.fetchone()
            if not result or result[0] < need:
                return False
            c.execute("UPDATE users SET credits = credits - ? WHERE user_id = ?", (need, user_id))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error checking credits: {e}")
        return False

def get_credits(user_id: str) -> str:
    """Get user's credit balance."""
    if not user_id:
        return "0"
    if user_id in ADMIN_USERS:
        return "∞"
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,))
            result = c.fetchone()
            return str(result[0]) if result else "0"
    except Exception as e:
        logger.error(f"Error getting credits: {e}")
        return "0"

def add_or_init_user(user_id: str, user_name: str = "User"):
    """Add or initialize user in database."""
    current_time = "2025-06-11 18:12:33"
    
    if "@" in user_id:
        default_credits = 75  # Email users get 75 credits
    else:
        default_credits = GUEST_INITIAL_CREDITS  # Guest users get 25 credits

    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            
            # Check if user exists
            c.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
            exists = c.fetchone()
            
            if not exists:
                c.execute(
                    """INSERT INTO users 
                    (user_id, user_name, credits, login_streak, last_login, last_guest_timestamp, last_reward_date) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        user_id,
                        user_name,
                        default_credits,
                        0,
                        current_time,
                        int(time.time()),
                        "",
                    ),
                )
                conn.commit()
                logger.info(f"New user initialized: {user_id} with {default_credits} credits")
            else:
                logger.info(f"User already exists: {user_id}")
                
    except Exception as e:
        logger.error(f"Error initializing user: {e}")
        raise

def save_chat_history(user_id: str, question: str, answer: str):
    """Save chat history to database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO chat_history (user_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
                (user_id, question, answer, "2025-06-11 18:12:33"),
            )
            conn.commit()
            logger.info(f"Chat history saved for user {user_id}")
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")
        raise

def get_chat_history(user_id: str, limit: int = 20) -> List[Dict[str, str]]:
    """Retrieve chat history for a user."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
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
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return []

# Configure OAuth
oauth = OAuth()
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    authorize_url="https://accounts.google.com/o/oauth2/auth",
    authorize_params=None,
    token_url="https://accounts.google.com/o/oauth2/token",
    token_params=None,
    api_base_url="https://www.googleapis.com/oauth2/v1/",
    client_kwargs={"scope": "openid email profile"}
)

# VirtuSim Service
class VirtuSimService:
    def __init__(self):
        self.api_key = VIRTUSIM_API_KEY
        self.base_url = "https://virtusim.com/api/v2/json.php"

    async def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make request to VirtuSim API with error handling."""
        try:
            if not self.api_key:
                logger.error("VIRTUSIM_API_KEY is not set")
                return {"status": False, "data": {"msg": "API key missing"}}

            params["api_key"] = self.api_key
            async with httpx.AsyncClient() as client:
                logger.info(f"VirtuSim Request - Params: {params}")
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                logger.info(f"VirtuSim Response - Status: {response.status_code}")
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"VirtuSim HTTP Error: {e}")
            return {"status": False, "data": {"msg": str(e)}}
        except Exception as e:
            logger.error(f"VirtuSim Error: {e}")
            return {"status": False, "data": {"msg": str(e)}}

    # Account Endpoints
    async def check_balance(self) -> Dict[str, Any]:
        """Check account balance."""
        return await self._make_request({"action": "balance"})

    async def get_balance_logs(self) -> Dict[str, Any]:
        """Get balance mutation history."""
        return await self._make_request({"action": "balance_logs"})

    async def get_recent_activity(self) -> Dict[str, Any]:
        """Get recent activity."""
        return await self._make_request({"action": "recent_activity"})

    # Service Endpoints
    async def get_available_services(self, country: str = "indonesia", service: str = "wa") -> Dict[str, Any]:
        """Get available services."""
        return await self._make_request({
            "action": "services",
            "country": country,
            "service": service
        })

    async def get_countries(self) -> Dict[str, Any]:
        """Get list of available countries."""
        return await self._make_request({"action": "list_country"})

    async def get_operators(self, country: str) -> Dict[str, Any]:
        """Get list of operators for a country."""
        return await self._make_request({
            "action": "list_operator",
            "country": country
        })

    # Transaction Endpoints
    async def get_active_orders(self) -> Dict[str, Any]:
        """Get active transactions."""
        return await self._make_request({"action": "active_order"})

    async def create_order(self, service: str, operator: str = "any") -> Dict[str, Any]:
        """Create new order."""
        return await self._make_request({
            "action": "order",
            "service": service,
            "operator": operator
        })

    async def reactive_order(self, order_id: str) -> Dict[str, Any]:
        """Reactivate existing order."""
        return await self._make_request({
            "action": "reactive_order",
            "id": order_id
        })

    async def check_order_status(self, order_id: str) -> Dict[str, Any]:
        """Check order status."""
        return await self._make_request({
            "action": "check_order",
            "id": order_id
        })

    async def set_order_status(self, order_id: str, status: int) -> Dict[str, Any]:
        """
        Change order status.
        Status codes:
        1 = Ready
        2 = Cancel
        3 = Resend
        4 = Completed
        """
        return await self._make_request({
            "action": "set_status",
            "id": order_id,
            "status": status
        })

    async def get_order_history(self) -> Dict[str, Any]:
        """Get order history."""
        return await self._make_request({"action": "order_history"})

    async def get_order_detail(self, order_id: str) -> Dict[str, Any]:
        """Get detailed order information."""
        return await self._make_request({
            "action": "detail_order",
            "id": order_id
        })

    async def create_deposit(self, method: int, amount: int, phone: str) -> Dict[str, Any]:
        """Create deposit request."""
        return await self._make_request({
            "action": "deposit",
            "method": method,
            "amount": amount,
            "phone": phone
        })

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

# API Endpoints
@app.get("/auth/google")
async def login_via_google(request: Request):
    """Initiate Google OAuth login."""
    redirect_uri = CALLBACK_URL
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

        add_or_init_user(email, user.get("name", email))
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

@app.post("/api/guest-login")
async def guest_login(request: Request):
    """Handle guest login."""
    try:
        data = await request.json()
        user_email = data.get("email")
        if not user_email:
            raise HTTPException(status_code=400, detail="Email wajib diisi")

        guest_id = f"guest_{int(time.time())}"
        add_or_init_user(user_email, guest_id)
        dummy_token = f"guest-token-{user_email.split('@')[0]}"
        credits = get_credits(user_email)
        
        logger.info(f"Guest login successful: {user_email} with {credits} credits")
        
        return JSONResponse(
            {
                "token": dummy_token,
                "credits": credits,
                "message": "Kugy.ai: Mode tamu aktif! 😺",
            }
        )
    except Exception as e:
        logger.error(f"Guest login error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# VirtuSim API Endpoints
@app.get("/api/virtusim/balance")
async def get_virtusim_balance():
    """Get VirtuSim balance."""
    return await virtusim_service.check_balance()

@app.get("/api/virtusim/balance/logs")
async def get_virtusim_balance_logs():
    """Get VirtuSim balance logs."""
    return await virtusim_service.get_balance_logs()

@app.get("/api/virtusim/activity")
async def get_virtusim_activity():
    """Get VirtuSim recent activity."""
    return await virtusim_service.get_recent_activity()

@app.get("/api/virtusim/services")
async def get_virtusim_services(
    country: str = Query("indonesia"),
    service: str = Query("wa")
):
    """Get VirtuSim services."""
    return await virtusim_service.get_available_services(country, service)

@app.get("/api/virtusim/countries")
async def get_virtusim_countries():
    """Get VirtuSim countries."""
    return await virtusim_service.get_countries()

@app.get("/api/virtusim/operators/{country}")
async def get_virtusim_operators(country: str):
    """Get VirtuSim operators for country."""
    return await virtusim_service.get_operators(country)

@app.get("/api/virtusim/orders/active")
async def get_virtusim_active_orders():
    """Get active VirtuSim orders."""
    return await virtusim_service.get_active_orders()

@app.post("/api/virtusim/orders")
async def create_virtusim_order(
    service: str = Query(...),
    operator: str = Query("any")
):
    """Create new VirtuSim order."""
    return await virtusim_service.create_order(service, operator)

@app.post("/api/virtusim/orders/{order_id}/reactive")
async def reactive_virtusim_order(order_id: str):
    """Reactivate VirtuSim order."""
    return await virtusim_service.reactive_order(order_id)

@app.get("/api/virtusim/orders/{order_id}/status")
async def check_virtusim_order_status(order_id: str):
    """Check VirtuSim order status."""
    return await virtusim_service.check_order_status(order_id)

@app.put("/api/virtusim/orders/{order_id}/status/{status}")
async def set_virtusim_order_status(order_id: str, status: int):
    """Set VirtuSim order status."""
    return await virtusim_service.set_order_status(order_id, status)

@app.get("/api/virtusim/orders/history")
async def get_virtusim_order_history():
    """Get VirtuSim order history."""
    return await virtusim_service.get_order_history()

@app.get("/api/virtusim/orders/{order_id}")
async def get_virtusim_order_detail(order_id: str):
    """Get VirtuSim order detail."""
    return await virtusim_service.get_order_detail(order_id)

@app.post("/api/virtusim/deposit")
async def create_virtusim_deposit(
    method: int = Query(...),
    amount: int = Query(...),
    phone: str = Query(...)
):
    """Create VirtuSim deposit."""
    return await virtusim_service.create_deposit(method, amount, phone)

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

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "status": "ok",
        "message": "API is running",
        "timestamp": "2025-06-11 18:12:33"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": "2025-06-11 18:12:33",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    try:
        ensure_db_and_log()  # Ensure DB and log exist before starting
        port = int(os.environ.get("PORT", 8080))
        uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise