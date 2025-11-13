from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
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

# Import and include routers (will be added later)
# from app.api import predictions, campaigns, results
# app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
# app.include_router(campaigns.router, prefix="/api/campaigns", tags=["campaigns"])
# app.include_router(results.router, prefix="/api/results", tags=["results"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
