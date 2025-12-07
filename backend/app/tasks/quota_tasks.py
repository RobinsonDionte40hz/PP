"""
Scheduled quota tasks for user quota management

This module contains Celery periodic tasks for:
- Resetting daily prediction quotas at midnight UTC
- Resetting monthly prediction quotas on the 1st of each month
"""
import logging
from celery import Task
from datetime import datetime, timezone

from celery_app import celery_app
from app.services.quota_service import quota_service

logger = logging.getLogger(__name__)


class QuotaTask(Task):
    """Base task for quota operations with error handling"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Log task failures"""
        logger.error(
            f"Quota task {task_id} failed: {exc}",
            exc_info=einfo
        )
    
    def on_success(self, retval, task_id, args, kwargs):
        """Log task success"""
        logger.info(
            f"Quota task {task_id} completed successfully: {retval}"
        )


@celery_app.task(base=QuotaTask, name="app.tasks.quota_tasks.reset_daily_quotas")
def reset_daily_quotas() -> dict:
    """
    Celery task to reset daily prediction quotas for all users
    
    This task should be scheduled to run at midnight UTC daily.
    It resets the daily_prediction_count to 0 for all users.
    
    Returns:
        Dictionary with reset statistics
    """
    logger.info("Starting daily quota reset...")
    
    try:
        users_updated = quota_service.reset_daily_quotas_all_users()
        
        result = {
            "status": "success",
            "users_updated": users_updated,
            "reset_at": datetime.now(timezone.utc).isoformat(),
            "reset_type": "daily"
        }
        
        logger.info(f"Daily quota reset complete: {users_updated} users updated")
        return result
        
    except Exception as e:
        logger.error(f"Error during daily quota reset: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "reset_at": datetime.now(timezone.utc).isoformat(),
            "reset_type": "daily"
        }


@celery_app.task(base=QuotaTask, name="app.tasks.quota_tasks.reset_monthly_quotas")
def reset_monthly_quotas() -> dict:
    """
    Celery task to reset monthly prediction quotas for all users
    
    This task should be scheduled to run at midnight UTC on the 1st of each month.
    It resets the monthly_prediction_count to 0 for all users.
    
    Returns:
        Dictionary with reset statistics
    """
    logger.info("Starting monthly quota reset...")
    
    try:
        users_updated = quota_service.reset_monthly_quotas_all_users()
        
        result = {
            "status": "success",
            "users_updated": users_updated,
            "reset_at": datetime.now(timezone.utc).isoformat(),
            "reset_type": "monthly"
        }
        
        logger.info(f"Monthly quota reset complete: {users_updated} users updated")
        return result
        
    except Exception as e:
        logger.error(f"Error during monthly quota reset: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "reset_at": datetime.now(timezone.utc).isoformat(),
            "reset_type": "monthly"
        }


@celery_app.task(base=QuotaTask, name="app.tasks.quota_tasks.check_and_reset_quotas")
def check_and_reset_quotas() -> dict:
    """
    Celery task to check if quota resets are needed and perform them
    
    This is a fallback task that can run more frequently (e.g., hourly)
    to catch any missed resets. It checks the current date/time and
    determines if daily or monthly resets are needed.
    
    Returns:
        Dictionary with reset statistics
    """
    logger.info("Checking if quota resets are needed...")
    
    now = datetime.now(timezone.utc)
    results = {
        "checked_at": now.isoformat(),
        "daily_reset": False,
        "monthly_reset": False
    }
    
    try:
        # Check if it's a new day (this is a safety net - individual user quotas
        # are also checked on-demand in the quota service)
        if now.hour == 0:  # Only reset during midnight hour
            logger.info("Midnight hour - triggering daily reset")
            daily_result = reset_daily_quotas()
            results["daily_reset"] = True
            results["daily_result"] = daily_result
        
        # Check if it's the 1st of the month at midnight
        if now.day == 1 and now.hour == 0:
            logger.info("First day of month at midnight - triggering monthly reset")
            monthly_result = reset_monthly_quotas()
            results["monthly_reset"] = True
            results["monthly_result"] = monthly_result
        
        return results
        
    except Exception as e:
        logger.error(f"Error during quota check: {e}", exc_info=True)
        results["error"] = str(e)
        return results
