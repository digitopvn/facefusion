import hashlib
import hmac
import os
import uuid as uuid_lib
from typing import Optional
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import redis.asyncio as redis
from dotenv import load_dotenv

load_dotenv()

# Configuration
SECURITY_CACHE_PREFIX = "security:nonce"
DURATION_CACHE = int(
    os.getenv("DURATION_CACHE_HASH_SECOND", "10800")
)  # 3 hours default
HASH_SECRET = os.getenv("HASH_SECRET", "your-secret-key-change-this")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Initialize Redis client
redis_client: Optional[redis.Redis] = None


async def get_redis_client() -> redis.Redis:
    """Get or create Redis client"""
    global redis_client
    if redis_client is None:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return redis_client


def make_hash(text: str) -> str:
    """
    Generate a hash from text using MD5 with pepper
    Matches the makeHash function in your Node.js code
    """
    try:
        # Add pepper (secret) to the text
        peppered_text = f"{text}{HASH_SECRET}"

        # Generate MD5 hash
        return hashlib.md5(peppered_text.encode("utf-8")).hexdigest()
    except Exception as e:
        raise ValueError(f"makeHash failed: {str(e)}")


def generate_hash() -> str:
    """
    Generate a new UUID and hash combination
    Matches the generateHash function in your Node.js code
    Returns: "uuid.hash" format
    """
    try:
        uuid = str(uuid_lib.uuid4())
        hash_value = make_hash(uuid)
        return f"{uuid}.{hash_value}"
    except Exception as e:
        raise ValueError(f"generateHash failed: {str(e)}")


async def validate_hash(hash_header: str) -> bool:
    """
    Validate the hash header format and value
    Returns True if valid, False otherwise
    """
    try:
        # Split the hash header into uuid and provided hash
        parts = hash_header.split(".")
        if len(parts) != 2:
            return False

        uuid, provided_hash = parts

        if not uuid or not provided_hash:
            return False

        # Generate expected hash
        expected_hash = make_hash(uuid)

        # Timing-safe comparison to prevent timing attacks
        hash_bytes = provided_hash.encode("utf-8")
        expected_bytes = expected_hash.encode("utf-8")

        if len(hash_bytes) != len(expected_bytes):
            return False

        # Use hmac.compare_digest for timing-safe comparison
        return hmac.compare_digest(hash_bytes, expected_bytes)

    except Exception:
        return False


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware to validate hash-based authentication
    Similar to requireHash in your Node.js implementation
    """

    def __init__(self, app, exclude_paths: list = None):  # type: ignore
        super().__init__(app)
        # Paths that don't require hash validation
        self.exclude_paths = exclude_paths or ["/docs", "/openapi.json", "/redoc"]

    async def dispatch(self, request: Request, call_next):
        # Skip validation for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)

        # Skip validation in test environment
        if os.getenv("NODE_ENV") == "test":
            return await call_next(request)

        # Skip validation for GET / (health check)
        if request.method == "GET" and request.url.path == "/":
            return await call_next(request)

        # Get hash header
        hash_header = request.headers.get("hash") or request.headers.get("Hash")

        if not hash_header:
            return JSONResponse(
                status_code=406, content={"status": 0, "message": "Missing Signature"}
            )

        # Validate hash
        is_valid = await validate_hash(hash_header)

        if not is_valid:
            return JSONResponse(
                status_code=406, content={"status": 0, "message": "Invalid Signature"}
            )

        # Optionally: Check Redis for nonce (to prevent replay attacks)
        # Uncomment the following code if you want to implement nonce checking
        try:
            redis_conn = await get_redis_client()
            uuid = hash_header.split(".")[0]
            cache_key = f"{SECURITY_CACHE_PREFIX}:{uuid}"

            # Check if nonce was already used
            used = await redis_conn.get(cache_key)
            if used:
                return JSONResponse(
                    status_code=406,
                    content={"status": 0, "message": "Nonce already used"},
                )

            # Mark nonce as used
            await redis_conn.setex(cache_key, DURATION_CACHE, "1")
        except Exception as e:
            # Log error but don't block request if Redis is down
            print(f"Redis error: {e}")

        # Continue with the request
        response = await call_next(request)
        return response
