"""Tasks package"""
# Legacy tasks (deprecated - use v2 versions)
from app.tasks.prediction_tasks import run_prediction, pause_prediction, stop_prediction

# V2 tasks using unified PredictionRunner (RECOMMENDED)
from app.tasks.prediction_tasks_v2 import (
    run_prediction_v2, 
    pause_prediction_v2, 
    stop_prediction_v2
)

# Screening tasks (aggregation risk assessment)
from app.tasks.screening_tasks import (
    run_batch_screening,
    run_screening_campaign,
)

# Quota tasks (usage limit management)
from app.tasks.quota_tasks import (
    reset_daily_quotas,
    reset_monthly_quotas,
    check_and_reset_quotas,
)

__all__ = [
    # Legacy (deprecated)
    "run_prediction", 
    "pause_prediction", 
    "stop_prediction",
    # V2 (recommended)
    "run_prediction_v2",
    "pause_prediction_v2", 
    "stop_prediction_v2",
    # Screening
    "run_batch_screening",
    "run_screening_campaign",
    # Quota management
    "reset_daily_quotas",
    "reset_monthly_quotas",
    "check_and_reset_quotas",
]
