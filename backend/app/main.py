from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import create_tables
from app.middleware import SecurityHeadersMiddleware, RequestLoggingMiddleware, CSRFMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Protein Prediction Platform API",
    description="API for managing protein structure predictions with security hardening",
    version="1.0.0",
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Create database tables
create_tables()
logger.info("Database tables created/verified")

# Security Middleware (order matters - apply from innermost to outermost)
# 1. Security headers (applied last, so added first)
app.add_middleware(SecurityHeadersMiddleware, enable_hsts=settings.ENABLE_HSTS)
logger.info("✓ Security headers middleware configured")

# 2. CSRF protection (Requirement 6.2)
if settings.ENABLE_CSRF:
    app.add_middleware(CSRFMiddleware, secret_key=settings.SECRET_KEY)
    logger.info("✓ CSRF protection middleware configured")

# 3. Request logging for security audit
app.add_middleware(RequestLoggingMiddleware)
logger.info("✓ Request logging middleware configured")

# Configure CORS (with environment-based origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-CSRF-Token", "X-Process-Time"],
)
logger.info(f"✓ CORS configured for origins: {settings.CORS_ORIGINS}")

@app.get("/")
async def root():
    return {"message": "Protein Prediction Platform API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "environment": settings.APP_ENV,
    }

# Import and include routers
from app.api import predictions, campaigns, results, sessions
from app.routes import websocket_routes, auth

app.include_router(auth.router)  # Authentication routes (/api/auth)
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(campaigns.router, prefix="/api/campaigns", tags=["campaigns"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(sessions.public_router, prefix="/api/shared", tags=["shared"])  # Public share links
app.include_router(websocket_routes.router, prefix="/api")  # WebSocket emission endpoints
app.include_router(results.router, prefix="/api/results", tags=["results"])

logger.info("✓ FastAPI app configured")
logger.info("✓ API routers mounted")

# Note: When running with uvicorn, use wsgi.py which wraps this app with Socket.IO
# Example: uvicorn wsgi:socket_app --reload --host 0.0.0.0 --port 8000

if __name__ == "__main__":
    # For development - run with Socket.IO wrapper
    import uvicorn
    logger.info("Starting server with Socket.IO support...")
    logger.info("Use 'uvicorn wsgi:socket_app --reload' for production")
    
    # Import and use the wrapped app from wsgi
    from wsgi import socket_app
    uvicorn.run(
        socket_app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
