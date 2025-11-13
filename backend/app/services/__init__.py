"""Services package"""
from app.services.prediction_service import prediction_service
from app.services.campaign_service import campaign_service

__all__ = ["prediction_service", "campaign_service"]
