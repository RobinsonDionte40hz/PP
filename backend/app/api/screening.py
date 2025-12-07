"""
Screening API endpoints for aggregation risk assessment.

This module provides endpoints for fast screening of protein sequences
for aggregation propensity - useful for filtering large sequence libraries.

Key Use Cases:
- Screen 100s-1000s of sequences before running full predictions
- Identify aggregation-prone candidates in therapeutic protein development
- Pre-filter peptide libraries
- Quality control for protein engineering

Unlike full structure prediction, screening is:
- FAST: 50-100 iterations vs 1000+
- FOCUSED: Answers "will it fold?" not "what's the structure?"
- SCALABLE: Batch processing with ranking/export
"""
from fastapi import APIRouter, HTTPException, Query, Path, BackgroundTasks, Request, Depends
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum
import logging
import os
import uuid

from app.security import require_auth_with_session, SecurityConfig
from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

# Rate limiting
IS_TESTING = os.getenv("TESTING", "false").lower() == "true"
limiter = Limiter(key_func=get_remote_address, enabled=not IS_TESTING)

router = APIRouter()


# ============================================================================
# Schemas
# ============================================================================

class ScreeningMode(str, Enum):
    """Screening speed/accuracy tradeoff."""
    FAST = "fast"           # 50 iterations, 2 agents - quickest
    BALANCED = "balanced"   # 100 iterations, 3 agents - default
    THOROUGH = "thorough"   # 200 iterations, 5 agents - most accurate


class AggregationRiskLevel(str, Enum):
    """Risk classification levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class SequenceScreeningRequest(BaseModel):
    """Request to screen a single sequence."""
    sequence: str = Field(..., min_length=5, max_length=500, description="Protein sequence")
    
    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        v = v.strip().upper()
        invalid = set(v) - SecurityConfig.VALID_AMINO_ACIDS
        if invalid:
            raise ValueError(f"Invalid amino acids: {', '.join(sorted(invalid))}")
        return v


class BatchScreeningRequest(BaseModel):
    """Request to screen multiple sequences."""
    sequences: List[str] = Field(..., description="List of sequences (1-500)")
    mode: ScreeningMode = Field(default=ScreeningMode.BALANCED, description="Screening speed mode")
    name: Optional[str] = Field(default=None, max_length=200, description="Optional batch name")
    
    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, v: List[str]) -> List[str]:
        if len(v) < 1:
            raise ValueError("At least 1 sequence required")
        if len(v) > 500:
            raise ValueError("Maximum 500 sequences allowed")
        validated = []
        for i, seq in enumerate(v):
            seq = seq.strip().upper()
            if len(seq) < 5:
                raise ValueError(f"Sequence {i+1} too short (minimum 5 amino acids)")
            if len(seq) > 500:
                raise ValueError(f"Sequence {i+1} too long (maximum 500 amino acids)")
            invalid = set(seq) - SecurityConfig.VALID_AMINO_ACIDS
            if invalid:
                raise ValueError(f"Sequence {i+1} has invalid amino acids: {', '.join(sorted(invalid))}")
            validated.append(seq)
        return validated


class SequenceScreeningResult(BaseModel):
    """Result for a single sequence screening."""
    sequence: str
    sequence_length: int
    
    # Scores (0-1, higher = better/lower risk)
    aggregation_score: float
    energy_score: float
    structure_score: float
    hydrophobic_score: float
    compactness_score: float
    
    # Classification
    risk_level: AggregationRiskLevel
    risk_factors: List[str]
    passes_screening: bool
    
    # Raw values
    final_energy: float
    secondary_structure_pct: float
    radius_of_gyration: float
    
    # Metadata
    screening_time_ms: float


class BatchScreeningResponse(BaseModel):
    """Response for batch screening."""
    batch_id: str
    name: Optional[str]
    mode: ScreeningMode
    status: str  # 'completed', 'running', 'failed'
    created_at: datetime
    completed_at: Optional[datetime]
    
    # Summary
    total_sequences: int
    sequences_passed: int
    sequences_failed: int
    
    # By risk level
    risk_summary: Dict[str, int]
    
    # Results (sorted by score, best first)
    results: List[SequenceScreeningResult]
    
    # Export paths (if saved)
    csv_path: Optional[str] = None
    json_path: Optional[str] = None


class ScreeningCampaignCreateRequest(BaseModel):
    """Create a screening campaign (ties into campaign system)."""
    name: str = Field(..., min_length=1, max_length=200)
    sequences: List[str] = Field(..., description="List of sequences (1-1000)")
    mode: ScreeningMode = Field(default=ScreeningMode.BALANCED)
    
    # Thresholds for filtering
    min_aggregation_score: float = Field(default=0.5, ge=0.0, le=1.0, 
        description="Minimum score to pass (0.5 = moderate risk allowed)")
    auto_create_predictions: bool = Field(default=False,
        description="Automatically create full predictions for sequences that pass")
    
    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, v: List[str]) -> List[str]:
        if len(v) < 1:
            raise ValueError("At least 1 sequence required")
        if len(v) > 1000:
            raise ValueError("Maximum 1000 sequences allowed")
        validated = []
        for i, seq in enumerate(v):
            seq = seq.strip().upper()
            if len(seq) < 5 or len(seq) > 500:
                raise ValueError(f"Sequence {i+1} must be 5-500 amino acids")
            invalid = set(seq) - SecurityConfig.VALID_AMINO_ACIDS
            if invalid:
                raise ValueError(f"Sequence {i+1} has invalid amino acids")
            validated.append(seq)
        return validated


class ScreeningCampaignResponse(BaseModel):
    """Response for screening campaign."""
    id: str
    name: str
    status: str
    mode: ScreeningMode
    created_at: datetime
    completed_at: Optional[datetime]
    
    # Progress
    total_sequences: int
    screened_sequences: int
    progress_percentage: float
    
    # Results summary
    passed_count: int
    failed_count: int
    risk_distribution: Dict[str, int]
    
    # Linked predictions (if auto_create_predictions was True)
    prediction_ids: List[str] = []
    
    # Export availability
    results_available: bool = False


# ============================================================================
# In-memory storage (would use database in production)
# ============================================================================

_screening_batches: Dict[str, BatchScreeningResponse] = {}
_screening_campaigns: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Helper Functions
# ============================================================================

def get_user_id(user: Dict[str, Any]) -> str:
    """Extract user_id from JWT token payload."""
    user_id = user.get("sub") or user.get("key_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    return user_id


def run_screening_sync(sequence: str, mode: ScreeningMode) -> SequenceScreeningResult:
    """
    Run screening synchronously (for single sequence or small batches).
    
    For larger batches, use the Celery task.
    """
    # Import from public API (SOLID: Dependency Inversion Principle)
    from ubf_protein.api import (
        AggregationScreener,
        ScreeningConfig,
        AggregationRisk as APIRisk,
    )
    
    # Map mode to config (window_size controls thoroughness)
    config_map = {
        ScreeningMode.FAST: ScreeningConfig(window_size=5, threshold=0.6),
        ScreeningMode.BALANCED: ScreeningConfig(window_size=7, threshold=0.5),
        ScreeningMode.THOROUGH: ScreeningConfig(window_size=9, threshold=0.4),
    }
    
    config = config_map.get(mode, config_map[ScreeningMode.BALANCED])
    screener = AggregationScreener()
    
    # Run screening
    result = screener.screen(sequence, config)
    
    # Map API risk level to endpoint enum
    risk_map = {
        APIRisk.LOW: AggregationRiskLevel.LOW,
        APIRisk.MODERATE: AggregationRiskLevel.MODERATE,
        APIRisk.HIGH: AggregationRiskLevel.HIGH,
        APIRisk.CRITICAL: AggregationRiskLevel.CRITICAL,
    }
    
    return SequenceScreeningResult(
        sequence=result.sequence,
        sequence_length=result.sequence_length,
        aggregation_score=result.aggregation_score,
        energy_score=result.energy_score,
        structure_score=result.structure_score,
        hydrophobic_score=result.hydrophobic_score,
        compactness_score=result.compactness_score,
        risk_level=risk_map[result.risk_level],
        risk_factors=result.risk_factors,
        passes_screening=result.passes_screening,
        final_energy=result.final_energy,
        secondary_structure_pct=result.secondary_structure_pct,
        radius_of_gyration=result.radius_of_gyration,
        screening_time_ms=result.screening_time_ms,
    )


# ============================================================================
# Endpoints
# ============================================================================

@router.post(
    "/single",
    response_model=SequenceScreeningResult,
    summary="Screen single sequence",
    description="Quick aggregation risk screening for a single sequence"
)
@limiter.limit("30/minute")
async def screen_single_sequence(
    request: Request,
    data: SequenceScreeningRequest,
    mode: ScreeningMode = Query(default=ScreeningMode.FAST, description="Screening mode"),
    user: Dict[str, Any] = Depends(require_auth_with_session),
):
    """
    Screen a single sequence for aggregation risk.
    
    This is a synchronous endpoint for quick checks. For batch processing,
    use the /batch endpoint.
    
    Returns immediately with screening results including:
    - Aggregation risk score (0-1, higher = lower risk)
    - Risk classification (low/moderate/high/critical)
    - Detailed metrics and risk factors
    """
    try:
        result = run_screening_sync(data.sequence, mode)
        return result
    except Exception as e:
        logger.error(f"Screening failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Screening failed: {str(e)}")


@router.post(
    "/batch",
    response_model=BatchScreeningResponse,
    status_code=202,
    summary="Screen batch of sequences",
    description="Screen multiple sequences for aggregation risk"
)
@limiter.limit("5/minute")
async def screen_batch(
    request: Request,
    data: BatchScreeningRequest,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(require_auth_with_session),
):
    """
    Submit a batch of sequences for aggregation screening.
    
    For small batches (≤10 sequences), results are returned immediately.
    For larger batches, processing happens in the background and you can
    poll the status endpoint.
    
    Results are sorted by aggregation_score (best candidates first).
    """
    user_id = get_user_id(user)
    batch_id = f"screen_{uuid.uuid4().hex[:12]}"
    
    # For small batches, run synchronously
    if len(data.sequences) <= 10:
        try:
            results = []
            for seq in data.sequences:
                result = run_screening_sync(seq, data.mode)
                results.append(result)
            
            # Sort by score (best first)
            results.sort(key=lambda x: -x.aggregation_score)
            
            # Calculate summary
            passed = sum(1 for r in results if r.passes_screening)
            risk_summary = {
                "low": sum(1 for r in results if r.risk_level == AggregationRiskLevel.LOW),
                "moderate": sum(1 for r in results if r.risk_level == AggregationRiskLevel.MODERATE),
                "high": sum(1 for r in results if r.risk_level == AggregationRiskLevel.HIGH),
                "critical": sum(1 for r in results if r.risk_level == AggregationRiskLevel.CRITICAL),
            }
            
            response = BatchScreeningResponse(
                batch_id=batch_id,
                name=data.name,
                mode=data.mode,
                status="completed",
                created_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                total_sequences=len(results),
                sequences_passed=passed,
                sequences_failed=len(results) - passed,
                risk_summary=risk_summary,
                results=results,
            )
            
            # Store for retrieval
            _screening_batches[batch_id] = response
            
            return response
            
        except Exception as e:
            logger.error(f"Batch screening failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Screening failed: {str(e)}")
    
    # For larger batches, queue background task
    try:
        # Create pending response
        response = BatchScreeningResponse(
            batch_id=batch_id,
            name=data.name,
            mode=data.mode,
            status="running",
            created_at=datetime.utcnow(),
            completed_at=None,
            total_sequences=len(data.sequences),
            sequences_passed=0,
            sequences_failed=0,
            risk_summary={"low": 0, "moderate": 0, "high": 0, "critical": 0},
            results=[],
        )
        
        _screening_batches[batch_id] = response
        
        # Queue Celery task
        try:
            from app.tasks import run_batch_screening
            task = run_batch_screening.delay(
                batch_id=batch_id,
                sequences=data.sequences,
                mode=data.mode.value,
                user_id=user_id,
            )
            logger.info(f"Queued batch screening {batch_id} with task {task.id}")
        except Exception as celery_error:
            logger.warning(f"Celery not available, running synchronously: {celery_error}")
            # Fall back to sync processing
            background_tasks.add_task(
                run_batch_screening_sync,
                batch_id,
                data.sequences,
                data.mode,
            )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to queue batch screening: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to start batch screening")


@router.get(
    "/batch/{batch_id}",
    response_model=BatchScreeningResponse,
    summary="Get batch screening status/results",
    description="Get status and results of a batch screening job"
)
async def get_batch_status(
    batch_id: str = Path(..., description="Batch ID"),
    user: Dict[str, Any] = Depends(require_auth_with_session),
):
    """Get the status and results of a batch screening job."""
    if batch_id not in _screening_batches:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    return _screening_batches[batch_id]


@router.get(
    "/batch/{batch_id}/export/csv",
    summary="Export batch results as CSV",
    description="Download batch screening results as CSV file"
)
async def export_batch_csv(
    batch_id: str = Path(..., description="Batch ID"),
    user: Dict[str, Any] = Depends(require_auth_with_session),
):
    """Export batch screening results as downloadable CSV."""
    from fastapi.responses import StreamingResponse
    import csv
    import io
    
    if batch_id not in _screening_batches:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    batch = _screening_batches[batch_id]
    if batch.status != "completed":
        raise HTTPException(status_code=400, detail="Batch not yet completed")
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        'rank', 'sequence', 'length', 'aggregation_score', 'risk_level',
        'energy_score', 'structure_score', 'hydrophobic_score', 'compactness_score',
        'final_energy', 'structure_pct', 'risk_factors', 'passes'
    ])
    
    # Data
    for rank, result in enumerate(batch.results, 1):
        writer.writerow([
            rank,
            result.sequence,
            result.sequence_length,
            f"{result.aggregation_score:.3f}",
            result.risk_level.value,
            f"{result.energy_score:.3f}",
            f"{result.structure_score:.3f}",
            f"{result.hydrophobic_score:.3f}",
            f"{result.compactness_score:.3f}",
            f"{result.final_energy:.2f}",
            f"{result.secondary_structure_pct:.1f}",
            '; '.join(result.risk_factors),
            'YES' if result.passes_screening else 'NO',
        ])
    
    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=screening_{batch_id}.csv"}
    )


@router.post(
    "/campaign",
    response_model=ScreeningCampaignResponse,
    status_code=202,
    summary="Create screening campaign",
    description="Create a screening campaign that filters sequences and optionally creates predictions"
)
@limiter.limit("3/minute")
async def create_screening_campaign(
    request: Request,
    data: ScreeningCampaignCreateRequest,
    background_tasks: BackgroundTasks,
    user: Dict[str, Any] = Depends(require_auth_with_session),
):
    """
    Create a screening campaign to filter sequences before full prediction.
    
    This is the recommended workflow for large sequence libraries:
    1. Submit sequences for screening
    2. Wait for screening to complete
    3. Review results - sequences are ranked by aggregation risk
    4. If auto_create_predictions=True, full predictions are automatically
       created for sequences that pass the min_aggregation_score threshold
    
    This saves compute time by not running expensive predictions on
    sequences that are likely to aggregate.
    """
    user_id = get_user_id(user)
    campaign_id = f"scamp_{uuid.uuid4().hex[:12]}"
    
    campaign = {
        "id": campaign_id,
        "name": data.name,
        "user_id": user_id,
        "status": "running",
        "mode": data.mode,
        "min_aggregation_score": data.min_aggregation_score,
        "auto_create_predictions": data.auto_create_predictions,
        "created_at": datetime.utcnow(),
        "completed_at": None,
        "total_sequences": len(data.sequences),
        "screened_sequences": 0,
        "passed_count": 0,
        "failed_count": 0,
        "risk_distribution": {"low": 0, "moderate": 0, "high": 0, "critical": 0},
        "prediction_ids": [],
        "results": [],
    }
    
    _screening_campaigns[campaign_id] = campaign
    
    # Queue processing
    try:
        from app.tasks import run_screening_campaign
        task = run_screening_campaign.delay(
            campaign_id=campaign_id,
            sequences=data.sequences,
            mode=data.mode.value,
            min_score=data.min_aggregation_score,
            auto_predict=data.auto_create_predictions,
            user_id=user_id,
        )
        logger.info(f"Queued screening campaign {campaign_id} with task {task.id}")
    except Exception as celery_error:
        logger.warning(f"Celery not available: {celery_error}")
        background_tasks.add_task(
            run_screening_campaign_sync,
            campaign_id,
            data.sequences,
            data.mode,
            data.min_aggregation_score,
            data.auto_create_predictions,
            user_id,
        )
    
    return ScreeningCampaignResponse(
        id=campaign_id,
        name=data.name,
        status="running",
        mode=data.mode,
        created_at=campaign["created_at"],
        completed_at=None,
        total_sequences=len(data.sequences),
        screened_sequences=0,
        progress_percentage=0.0,
        passed_count=0,
        failed_count=0,
        risk_distribution=campaign["risk_distribution"],
        prediction_ids=[],
        results_available=False,
    )


@router.get(
    "/campaign/{campaign_id}",
    response_model=ScreeningCampaignResponse,
    summary="Get screening campaign status",
    description="Get status and progress of a screening campaign"
)
async def get_screening_campaign(
    campaign_id: str = Path(..., description="Campaign ID"),
    user: Dict[str, Any] = Depends(require_auth_with_session),
):
    """Get the status of a screening campaign."""
    if campaign_id not in _screening_campaigns:
        raise HTTPException(status_code=404, detail="Screening campaign not found")
    
    campaign = _screening_campaigns[campaign_id]
    
    progress = 0.0
    if campaign["total_sequences"] > 0:
        progress = (campaign["screened_sequences"] / campaign["total_sequences"]) * 100
    
    return ScreeningCampaignResponse(
        id=campaign["id"],
        name=campaign["name"],
        status=campaign["status"],
        mode=campaign["mode"],
        created_at=campaign["created_at"],
        completed_at=campaign.get("completed_at"),
        total_sequences=campaign["total_sequences"],
        screened_sequences=campaign["screened_sequences"],
        progress_percentage=progress,
        passed_count=campaign["passed_count"],
        failed_count=campaign["failed_count"],
        risk_distribution=campaign["risk_distribution"],
        prediction_ids=campaign.get("prediction_ids", []),
        results_available=campaign["status"] == "completed",
    )


@router.get(
    "/campaign/{campaign_id}/results",
    summary="Get screening campaign results",
    description="Get detailed results of a completed screening campaign"
)
async def get_screening_campaign_results(
    campaign_id: str = Path(..., description="Campaign ID"),
    passed_only: bool = Query(default=False, description="Only return sequences that passed"),
    user: Dict[str, Any] = Depends(require_auth_with_session),
):
    """Get detailed results from a screening campaign."""
    if campaign_id not in _screening_campaigns:
        raise HTTPException(status_code=404, detail="Screening campaign not found")
    
    campaign = _screening_campaigns[campaign_id]
    
    if campaign["status"] != "completed":
        raise HTTPException(status_code=400, detail="Campaign not yet completed")
    
    results = campaign.get("results", [])
    
    if passed_only:
        results = [r for r in results if r.get("passes_screening", False)]
    
    return {
        "campaign_id": campaign_id,
        "total_results": len(results),
        "results": results,
    }


# ============================================================================
# Background task helpers (sync fallback when Celery unavailable)
# ============================================================================

async def run_batch_screening_sync(batch_id: str, sequences: List[str], mode: ScreeningMode):
    """Run batch screening synchronously (fallback)."""
    try:
        results = []
        for seq in sequences:
            result = run_screening_sync(seq, mode)
            results.append(result)
        
        results.sort(key=lambda x: -x.aggregation_score)
        
        passed = sum(1 for r in results if r.passes_screening)
        risk_summary = {
            "low": sum(1 for r in results if r.risk_level == AggregationRiskLevel.LOW),
            "moderate": sum(1 for r in results if r.risk_level == AggregationRiskLevel.MODERATE),
            "high": sum(1 for r in results if r.risk_level == AggregationRiskLevel.HIGH),
            "critical": sum(1 for r in results if r.risk_level == AggregationRiskLevel.CRITICAL),
        }
        
        _screening_batches[batch_id].status = "completed"
        _screening_batches[batch_id].completed_at = datetime.utcnow()
        _screening_batches[batch_id].results = results
        _screening_batches[batch_id].sequences_passed = passed
        _screening_batches[batch_id].sequences_failed = len(results) - passed
        _screening_batches[batch_id].risk_summary = risk_summary
        
    except Exception as e:
        logger.error(f"Batch screening failed: {e}", exc_info=True)
        _screening_batches[batch_id].status = "failed"


async def run_screening_campaign_sync(
    campaign_id: str,
    sequences: List[str],
    mode: ScreeningMode,
    min_score: float,
    auto_predict: bool,
    user_id: str,
):
    """Run screening campaign synchronously (fallback)."""
    campaign = _screening_campaigns[campaign_id]
    
    try:
        results = []
        for i, seq in enumerate(sequences):
            result = run_screening_sync(seq, mode)
            results.append(result.model_dump())
            
            # Update progress
            campaign["screened_sequences"] = i + 1
            
            # Update risk distribution
            risk = result.risk_level.value
            campaign["risk_distribution"][risk] = campaign["risk_distribution"].get(risk, 0) + 1
            
            if result.passes_screening and result.aggregation_score >= min_score:
                campaign["passed_count"] += 1
            else:
                campaign["failed_count"] += 1
        
        # Sort by score
        results.sort(key=lambda x: -x["aggregation_score"])
        campaign["results"] = results
        
        # Auto-create predictions for passed sequences if requested
        if auto_predict:
            passed_seqs = [
                r["sequence"] for r in results 
                if r.get("passes_screening") and r.get("aggregation_score", 0) >= min_score
            ]
            
            # Create predictions for passed sequences
            # This would integrate with prediction_service
            logger.info(f"Would create {len(passed_seqs)} predictions for passed sequences")
            # TODO: Implement auto-prediction creation
        
        campaign["status"] = "completed"
        campaign["completed_at"] = datetime.utcnow()
        
    except Exception as e:
        logger.error(f"Screening campaign failed: {e}", exc_info=True)
        campaign["status"] = "failed"
