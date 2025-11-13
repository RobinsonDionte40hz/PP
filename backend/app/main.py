from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import create_tables
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Protein Prediction Platform API",
    description="API for managing protein structure predictions",
    version="1.0.0",
)

# Create database tables
create_tables()
logger.info("Database tables created/verified")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React default
        "http://localhost:5173",  # Vite default
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
from app.api import predictions, campaigns, results
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(campaigns.router, prefix="/api/campaigns", tags=["campaigns"])
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
