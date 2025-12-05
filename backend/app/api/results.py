"""
Results API endpoints
"""
from fastapi import APIRouter, HTTPException, Path, Query, Response
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path as FilePath
from typing import Optional, Dict, Any
from app.services.prediction_service import prediction_service
import logging
import json
import os

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/{prediction_id}",
    summary="Get detailed results",
    description="Get comprehensive results for a completed prediction"
)
async def get_results(
    prediction_id: str = Path(..., description="Prediction ID")
):
    """
    Get detailed results for a prediction.
    
    Includes metrics, structure information, and analysis.
    """
    from pathlib import Path as FilePath
    import json
    
    prediction = prediction_service.get_prediction(prediction_id)
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    # Check if results exist
    if not prediction.result_path:
        raise HTTPException(status_code=404, detail="Results not available yet")
    
    # Load result file
    result_dir = FilePath(prediction.result_path)
    result_file = result_dir / "results.json"
    
    if not result_file.exists():
        # Return basic info if detailed results not yet saved
        return {
            "prediction_id": prediction_id,
            "status": prediction.status.value,
            "metrics": prediction.metrics,
            "result_path": prediction.result_path
        }
    
    # Load detailed results
    with open(result_file, 'r') as f:
        results_data = json.load(f)
    
    return {
        "prediction_id": prediction_id,
        "status": prediction.status.value,
        "sequence": prediction.sequence,
        "configuration": prediction.configuration,
        "results": results_data,
        "result_path": prediction.result_path,
        "checkpoint_path": prediction.checkpoint_path
    }


@router.get(
    "/{prediction_id}/structure",
    summary="Get PDB structure",
    description="Download the predicted structure in PDB format"
)
async def get_structure(
    prediction_id: str = Path(..., description="Prediction ID")
):
    """
    Download the predicted protein structure in PDB format.
    """
    from pathlib import Path as FilePath
    from fastapi.responses import FileResponse
    
    prediction = prediction_service.get_prediction(prediction_id)
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    if not prediction.result_path:
        raise HTTPException(status_code=404, detail="Results not available yet")
    
    # Look for PDB file
    result_dir = FilePath(prediction.result_path)
    pdb_file = result_dir / "structure.pdb"
    
    if not pdb_file.exists():
        raise HTTPException(status_code=404, detail="PDB structure file not found")
    
    return FileResponse(
        path=str(pdb_file),
        media_type="chemical/x-pdb",
        filename=f"{prediction_id}.pdb"
    )


@router.get(
    "/{prediction_id}/trajectory",
    summary="Get trajectory data",
    description="Get trajectory data showing exploration path"
)
async def get_trajectory(
    prediction_id: str = Path(..., description="Prediction ID")
):
    """
    Get trajectory data for visualization.
    
    Returns coordinates and energies along the exploration path.
    """
    import json
    
    prediction = prediction_service.get_prediction(prediction_id)
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    # Load trajectory JSON from result path
    if not prediction.result_path:
        return {
            "prediction_id": prediction_id,
            "total_points": 0,
            "agent_count": 0,
            "trajectory": [],
            "message": "No trajectory data available - prediction may still be running"
        }
    
    trajectory_file = FilePath(prediction.result_path) / "trajectory.json"
    
    if not trajectory_file.exists():
        return {
            "prediction_id": prediction_id,
            "total_points": 0,
            "agent_count": 0,
            "trajectory": [],
            "message": "Trajectory file not found - prediction may have been run before trajectory tracking was enabled"
        }
    
    try:
        with open(trajectory_file, 'r') as f:
            trajectory_data = json.load(f)
        return trajectory_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load trajectory data: {str(e)}")


@router.get(
    "/{prediction_id}/geometric",
    summary="Get geometric analysis",
    description="Get geometric attractor analysis including phi angles and patterns"
)
async def get_geometric_analysis(
    prediction_id: str = Path(..., description="Prediction ID")
):
    """
    Get geometric analysis data for visualization.
    
    Returns phi angle distributions, Platonic solid patterns, and QCPP metrics.
    """
    prediction = prediction_service.get_prediction(prediction_id)
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    if not prediction.result_path:
        return {
            "prediction_id": prediction_id,
            "message": "No results available - prediction may still be running",
            "geometric_analysis": None
        }
    
    result_file = FilePath(prediction.result_path) / "results.json"
    
    if not result_file.exists():
        return {
            "prediction_id": prediction_id,
            "message": "Results file not found",
            "geometric_analysis": None
        }
    
    try:
        with open(result_file, 'r') as f:
            results_data = json.load(f)
        
        geometric = results_data.get('geometric_analysis')
        qcpp_metrics = {
            'qaap_alignment': results_data.get('qaap_alignment'),
            'resonance_40hz': results_data.get('resonance_40hz'),
            'water_shielding': results_data.get('water_shielding'),
            'qcp_score': results_data.get('qcp_score'),
            'qcpp_cache_hit_rate': results_data.get('qcpp_cache_hit_rate'),
        }
        
        return {
            "prediction_id": prediction_id,
            "geometric_analysis": geometric,
            "qcpp_metrics": qcpp_metrics,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load results: {str(e)}")


@router.get(
    "/{prediction_id}/metrics",
    summary="Get detailed metrics",
    description="Get comprehensive metrics breakdown"
)
async def get_metrics(
    prediction_id: str = Path(..., description="Prediction ID")
):
    """
    Get detailed metrics for the prediction.
    
    Includes energy components, RMSD evolution, agent statistics, etc.
    """
    prediction = prediction_service.get_prediction(prediction_id)
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    return {
        "prediction_id": prediction_id,
        "metrics": prediction.metrics,
        "current_iteration": prediction.current_iteration,
        "total_iterations": prediction.total_iterations,
        "message": "Extended metrics loading not implemented yet"
    }


@router.get(
    "/{prediction_id}/export",
    summary="Export results",
    description="Export results in various formats (JSON, PDF, ZIP)"
)
async def export_results(
    prediction_id: str = Path(..., description="Prediction ID"),
    format: str = Query("json", description="Export format: json, csv, zip")
):
    """
    Export prediction results in the specified format.
    
    - json: Complete results as JSON
    - csv: Metrics as CSV
    - zip: Full package (PDB, trajectory, metrics, plots)
    """
    prediction = prediction_service.get_prediction(prediction_id)
    
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    if format not in ["json", "csv", "zip"]:
        raise HTTPException(status_code=400, detail="Invalid format. Use: json, csv, or zip")
    
    # TODO: Implement export functionality
    return {
        "message": f"Export in {format} format not implemented yet",
        "prediction_id": prediction_id,
        "format": format
    }


@router.post(
    "/compare",
    summary="Compare results",
    description="Compare multiple prediction results"
)
async def compare_results(
    prediction_ids: list[str] = Query(..., description="List of prediction IDs to compare")
):
    """
    Compare multiple predictions side-by-side.
    
    Returns comparative metrics and statistics.
    """
    if len(prediction_ids) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 predictions to compare")
    
    if len(prediction_ids) > 10:
        raise HTTPException(status_code=400, detail="Cannot compare more than 10 predictions")
    
    # Get all predictions
    predictions = []
    for pred_id in prediction_ids:
        pred = prediction_service.get_prediction(pred_id)
        if pred:
            predictions.append(pred)
    
    if len(predictions) != len(prediction_ids):
        raise HTTPException(status_code=404, detail="One or more predictions not found")
    
    # TODO: Implement comparison logic
    comparison_data = {
        "predictions": [
            {
                "id": p.id,
                "sequence": p.sequence,
                "status": p.status.value,
                "metrics": p.metrics
            }
            for p in predictions
        ],
        "message": "Detailed comparison not implemented yet"
    }
    
    return comparison_data
