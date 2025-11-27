"""
Scheduled cleanup tasks for session maintenance

This module contains Celery periodic tasks for:
- Cleaning up expired work sessions
- Removing expired share links
"""
import logging
from celery import Task

from celery_app import celery_app
from app.services.session_cleanup_service import get_cleanup_service

logger = logging.getLogger(__name__)


class CleanupTask(Task):
    """Base task for cleanup operations with error handling"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Log task failures"""
        logger.error(
            f"Cleanup task {task_id} failed: {exc}",
            exc_info=einfo
        )
    
    def on_success(self, retval, task_id, args, kwargs):
        """Log task success"""
        logger.info(
            f"Cleanup task {task_id} completed successfully: {retval}"
        )


@celery_app.task(base=CleanupTask, name="app.tasks.cleanup_tasks.cleanup_expired_sessions_task")
def cleanup_expired_sessions_task(retention_days: int | None = None) -> dict:
    """
    Celery task to clean up expired sessions
    
    This task:
    1. Identifies sessions inactive for > retention_days
    2. Deletes session records from database
    3. Removes session directories from file system
    4. Cleans up associated shared exports
    
    Args:
        retention_days: Number of days to retain inactive sessions
                       (defaults to settings.SESSION_RETENTION_DAYS)
    
    Returns:
        Dictionary with cleanup statistics
        
    Example (manual execution):
        >>> from app.tasks.cleanup_tasks import cleanup_expired_sessions_task
        >>> result = cleanup_expired_sessions_task(retention_days=90)
        >>> print(f"Deleted {result['sessions_deleted']} sessions")
    """
    from app.config import settings
    
    if retention_days is None:
        retention_days = settings.SESSION_RETENTION_DAYS
    
    logger.info(f"Starting session cleanup (retention_days={retention_days})")
    
    cleanup_service = get_cleanup_service()
    stats = cleanup_service.delete_expired_sessions(
        retention_days=retention_days,
        dry_run=False
    )
    
    # Also cleanup expired share links
    share_stats = cleanup_service.cleanup_expired_shares()
    stats["shares_deleted"] = share_stats["shares_deleted"]
    stats["share_errors"] = share_stats["errors"]
    
    logger.info(
        f"Cleanup completed: {stats['sessions_deleted']} sessions, "
        f"{stats['shares_deleted']} shares, "
        f"{len(stats['errors']) + len(stats['share_errors'])} errors"
    )
    
    return stats


@celery_app.task(base=CleanupTask, name="app.tasks.cleanup_tasks.cleanup_expired_shares_task")
def cleanup_expired_shares_task() -> dict:
    """
    Celery task to clean up expired share links only
    
    This removes SharedExport records where expires_at has passed,
    without affecting the associated sessions.
    
    Returns:
        Dictionary with cleanup statistics
        
    Example (manual execution):
        >>> from app.tasks.cleanup_tasks import cleanup_expired_shares_task
        >>> result = cleanup_expired_shares_task()
        >>> print(f"Deleted {result['shares_deleted']} expired shares")
    """
    logger.info("Starting share link cleanup")
    
    cleanup_service = get_cleanup_service()
    stats = cleanup_service.cleanup_expired_shares()
    
    logger.info(f"Share cleanup completed: {stats['shares_deleted']} shares deleted")
    
    return stats
