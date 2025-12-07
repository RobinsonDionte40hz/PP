from celery import Celery
from celery.schedules import crontab
from app.config import settings

celery_app = Celery(
    "pp_tasks",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "app.tasks.prediction_tasks",
        "app.tasks.prediction_tasks_v2",
        "app.tasks.cleanup_tasks",
        "app.tasks.quota_tasks"
    ]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_send_sent_event=True,
)

# Periodic task schedule
celery_app.conf.beat_schedule = {
    "cleanup-expired-sessions": {
        "task": "app.tasks.cleanup_tasks.cleanup_expired_sessions_task",
        "schedule": crontab(hour=2, minute=0),  # Daily at 2 AM UTC
        "kwargs": {"retention_days": settings.SESSION_RETENTION_DAYS},
    },
    "cleanup-expired-shares": {
        "task": "app.tasks.cleanup_tasks.cleanup_expired_shares_task",
        "schedule": crontab(hour=3, minute=0),  # Daily at 3 AM UTC
    },
    # Quota management tasks
    "reset-daily-quotas": {
        "task": "app.tasks.quota_tasks.reset_daily_quotas",
        "schedule": crontab(hour=0, minute=0),  # Daily at midnight UTC
    },
    "reset-monthly-quotas": {
        "task": "app.tasks.quota_tasks.reset_monthly_quotas",
        "schedule": crontab(hour=0, minute=5, day_of_month=1),  # 1st of month at 00:05 UTC
    },
}
