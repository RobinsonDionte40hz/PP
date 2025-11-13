# API Documentation - Protein Prediction Platform

This document describes the REST API endpoints for the Protein Prediction Platform.

## Table of Contents

- [Base URL](#base-url)
- [Authentication](#authentication)
- [Prediction Endpoints](#prediction-endpoints)
- [Campaign Endpoints](#campaign-endpoints)
- [Results Endpoints](#results-endpoints)
- [WebSocket Events](#websocket-events)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)

## Base URL

**Development**: `http://localhost:8000`
**Production**: `https://your-domain.com`

All API endpoints are prefixed with `/api/v1` unless otherwise specified.

## Authentication

Currently, the API does not require authentication. For production deployments, implement JWT or session-based authentication.

## Interactive Documentation

FastAPI provides interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These interfaces allow you to test endpoints directly in your browser.

## Prediction Endpoints

### Create Prediction

Submit a new protein structure prediction job.

**Endpoint**: `POST /api/predictions`

**Request Body**:
```json
{
  "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
  "iterations": 1000,
  "agents": 10,
  "enable_qcpp": true,
  "enable_mediators": true,
  "enable_geometric_targeting": true,
  "enable_refinement": true,
  "qcpp_config": "default",
  "diversity_strategy": "balanced"
}
```

**Parameters**:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `sequence` | string | Yes | - | Protein amino acid sequence (1-letter code) |
| `iterations` | integer | No | 1000 | Number of optimization iterations |
| `agents` | integer | No | 10 | Number of agents for exploration |
| `enable_qcpp` | boolean | No | true | Enable QCPP integration |
| `enable_mediators` | boolean | No | false | Enable mediator agents |
| `enable_geometric_targeting` | boolean | No | false | Enable geometric attractor targeting |
| `enable_refinement` | boolean | No | false | Enable quantum refinement |
| `qcpp_config` | string | No | "default" | QCPP config preset: default/high_performance/high_accuracy |
| `diversity_strategy` | string | No | "balanced" | Agent diversity: cautious/balanced/aggressive/mixed |

**Response** (201 Created):
```json
{
  "id": "pred_abc123",
  "status": "queued",
  "sequence": "MQIFVKT...",
  "config": {
    "iterations": 1000,
    "agents": 10,
    "enable_qcpp": true,
    "enable_mediators": true
  },
  "created_at": "2025-11-12T10:30:00Z",
  "estimated_completion": "2025-11-12T10:45:00Z"
}
```

**Error Responses**:
- `400 Bad Request`: Invalid sequence or parameters
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

---

### Get Prediction Details

Retrieve details of a specific prediction.

**Endpoint**: `GET /api/predictions/{prediction_id}`

**Response** (200 OK):
```json
{
  "id": "pred_abc123",
  "status": "running",
  "sequence": "MQIFVKT...",
  "progress": {
    "current_iteration": 450,
    "total_iterations": 1000,
    "percentage": 45.0,
    "estimated_time_remaining": "00:08:30"
  },
  "metrics": {
    "current_energy": -156.3,
    "best_energy": -189.2,
    "current_rmsd": 8.5,
    "best_rmsd": 7.2
  },
  "config": { ... },
  "created_at": "2025-11-12T10:30:00Z",
  "started_at": "2025-11-12T10:30:15Z",
  "completed_at": null
}
```

**Status Values**:
- `queued`: Waiting to start
- `running`: Currently executing
- `paused`: Temporarily paused
- `completed`: Successfully finished
- `failed`: Execution failed
- `cancelled`: User cancelled

---

### List Predictions

Get a list of all predictions with optional filtering.

**Endpoint**: `GET /api/predictions`

**Query Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `status` | string | Filter by status: queued/running/completed/failed |
| `limit` | integer | Max results to return (default: 50, max: 100) |
| `offset` | integer | Offset for pagination (default: 0) |
| `sort_by` | string | Sort field: created_at/updated_at (default: created_at) |
| `order` | string | Sort order: asc/desc (default: desc) |

**Example**: `GET /api/predictions?status=completed&limit=10`

**Response** (200 OK):
```json
{
  "items": [
    {
      "id": "pred_abc123",
      "status": "completed",
      "sequence": "MQIFVKT...",
      "created_at": "2025-11-12T10:30:00Z",
      "completed_at": "2025-11-12T10:45:23Z",
      "metrics": {
        "best_energy": -189.2,
        "best_rmsd": 7.2
      }
    }
  ],
  "total": 45,
  "limit": 10,
  "offset": 0
}
```

---

### Pause Prediction

Pause a running prediction.

**Endpoint**: `POST /api/predictions/{prediction_id}/pause`

**Response** (200 OK):
```json
{
  "id": "pred_abc123",
  "status": "paused",
  "message": "Prediction paused successfully"
}
```

---

### Resume Prediction

Resume a paused prediction.

**Endpoint**: `POST /api/predictions/{prediction_id}/resume`

**Response** (200 OK):
```json
{
  "id": "pred_abc123",
  "status": "running",
  "message": "Prediction resumed successfully"
}
```

---

### Stop Prediction

Stop a running prediction permanently.

**Endpoint**: `POST /api/predictions/{prediction_id}/stop`

**Response** (200 OK):
```json
{
  "id": "pred_abc123",
  "status": "cancelled",
  "message": "Prediction stopped successfully"
}
```

---

### Delete Prediction

Delete a prediction and its associated data.

**Endpoint**: `DELETE /api/predictions/{prediction_id}`

**Response** (204 No Content)

---

### Download Checkpoint

Download the checkpoint file for a prediction.

**Endpoint**: `GET /api/predictions/{prediction_id}/checkpoint`

**Response** (200 OK):
- Content-Type: `application/json`
- Downloads checkpoint JSON file

---

## Campaign Endpoints

### Create Campaign

Create a systematic testing campaign for multiple proteins.

**Endpoint**: `POST /api/campaigns`

**Request Body**:
```json
{
  "name": "Test Campaign 1",
  "proteins": ["1UBQ", "1CRN", "2MR9"],
  "configurations": [
    {
      "name": "Base Optimal",
      "iterations": 1000,
      "agents": 10,
      "enable_qcpp": true
    },
    {
      "name": "High Iterations",
      "iterations": 5000,
      "agents": 10,
      "enable_qcpp": true
    }
  ],
  "quality_gates": {
    "min_rmsd": 10.0,
    "max_energy": 0.0
  }
}
```

**Response** (201 Created):
```json
{
  "id": "camp_xyz789",
  "name": "Test Campaign 1",
  "status": "queued",
  "total_proteins": 3,
  "total_configs": 2,
  "total_predictions": 6,
  "current_phase": 0,
  "created_at": "2025-11-12T11:00:00Z"
}
```

---

### Get Campaign Details

**Endpoint**: `GET /api/campaigns/{campaign_id}`

**Response** (200 OK):
```json
{
  "id": "camp_xyz789",
  "name": "Test Campaign 1",
  "status": "running",
  "progress": {
    "completed": 2,
    "total": 6,
    "percentage": 33.3,
    "current_protein": "1CRN"
  },
  "statistics": {
    "avg_rmsd": 8.4,
    "avg_energy": -156.8,
    "success_rate": 66.7,
    "total_duration": "01:23:45"
  },
  "created_at": "2025-11-12T11:00:00Z",
  "started_at": "2025-11-12T11:00:30Z"
}
```

---

### List Campaigns

**Endpoint**: `GET /api/campaigns`

Query parameters same as predictions list endpoint.

---

### Get Campaign Statistics

**Endpoint**: `GET /api/campaigns/{campaign_id}/statistics`

**Response** (200 OK):
```json
{
  "campaign_id": "camp_xyz789",
  "overall": {
    "total_predictions": 6,
    "completed": 5,
    "failed": 1,
    "avg_rmsd": 8.4,
    "avg_energy": -156.8,
    "best_rmsd": 7.2,
    "best_energy": -189.2
  },
  "by_protein": {
    "1UBQ": {
      "predictions": 2,
      "avg_rmsd": 7.8,
      "avg_energy": -165.3
    }
  },
  "by_config": {
    "Base Optimal": {
      "predictions": 3,
      "avg_rmsd": 8.9,
      "avg_energy": -145.6
    }
  }
}
```

---

### Resume Campaign

**Endpoint**: `POST /api/campaigns/{campaign_id}/resume`

---

### Delete Campaign

**Endpoint**: `DELETE /api/campaigns/{campaign_id}`

---

## Results Endpoints

### Get Results

Get detailed results for a completed prediction.

**Endpoint**: `GET /api/results/{prediction_id}`

**Response** (200 OK):
```json
{
  "prediction_id": "pred_abc123",
  "status": "completed",
  "summary": {
    "best_energy": -189.2,
    "best_rmsd": 7.2,
    "total_iterations": 1000,
    "duration": "00:15:23",
    "quality_score": 0.85
  },
  "structure": {
    "residues": 76,
    "bonds": 150,
    "secondary_structure": {
      "helix": 35,
      "sheet": 20,
      "coil": 21
    }
  },
  "energy_breakdown": {
    "bond": -45.2,
    "angle": -32.1,
    "dihedral": -28.5,
    "vdw": -52.3,
    "electrostatic": -21.8,
    "hbond": -9.3
  },
  "geometric_analysis": {
    "patterns_detected": ["icosahedron", "golden_spiral"],
    "phi_ratio_score": 0.92
  },
  "agent_statistics": {
    "total_moves": 15430,
    "accepted_moves": 8921,
    "acceptance_rate": 0.578
  }
}
```

---

### Get Structure (PDB)

Download the predicted structure in PDB format.

**Endpoint**: `GET /api/results/{prediction_id}/structure`

**Response** (200 OK):
- Content-Type: `chemical/x-pdb`
- Downloads PDB file

---

### Get Trajectory

Get trajectory data for visualization.

**Endpoint**: `GET /api/results/{prediction_id}/trajectory`

**Query Parameters**:
- `interval` (integer): Sample interval (default: 10)
- `max_points` (integer): Maximum data points (default: 1000)

**Response** (200 OK):
```json
{
  "prediction_id": "pred_abc123",
  "iterations": [0, 10, 20, 30, ...],
  "energy": [-120.5, -135.2, -148.9, ...],
  "rmsd": [12.3, 11.8, 10.5, ...],
  "parameters": {
    "aggressiveness": [8.0, 8.2, 7.9, ...],
    "consistency": [0.65, 0.68, 0.71, ...]
  }
}
```

---

### Get Detailed Metrics

**Endpoint**: `GET /api/results/{prediction_id}/metrics`

Returns comprehensive metrics including per-agent statistics, memory usage, quantum analysis, etc.

---

### Export Results

Export results in various formats.

**Endpoint**: `GET /api/results/{prediction_id}/export`

**Query Parameters**:
- `format` (string): json/pdf/csv (default: json)

**Response**:
- Downloads file in requested format

---

### Compare Results

Compare multiple predictions.

**Endpoint**: `POST /api/results/compare`

**Request Body**:
```json
{
  "prediction_ids": ["pred_abc123", "pred_def456", "pred_ghi789"]
}
```

**Response** (200 OK):
```json
{
  "predictions": [
    {
      "id": "pred_abc123",
      "energy": -189.2,
      "rmsd": 7.2,
      "quality_score": 0.85
    }
  ],
  "comparison": {
    "best_energy": "pred_abc123",
    "best_rmsd": "pred_def456",
    "best_overall": "pred_abc123"
  }
}
```

---

## WebSocket Events

Connect to WebSocket for real-time updates.

**URL**: `ws://localhost:8000/socket.io/`

**Protocol**: Socket.IO

### Client Events (Emit)

#### Join Prediction Room

```javascript
socket.emit('join_prediction', {
  prediction_id: 'pred_abc123'
});
```

#### Leave Prediction Room

```javascript
socket.emit('leave_prediction', {
  prediction_id: 'pred_abc123'
});
```

### Server Events (Listen)

#### Progress Update

Sent every iteration or at specified intervals.

```javascript
socket.on('progress_update', (data) => {
  console.log(data);
  // {
  //   prediction_id: 'pred_abc123',
  //   iteration: 450,
  //   total_iterations: 1000,
  //   percentage: 45.0,
  //   metrics: {
  //     current_energy: -156.3,
  //     current_rmsd: 8.5
  //   }
  // }
});
```

#### Agent Update

Sent when agent states change.

```javascript
socket.on('agent_update', (data) => {
  // {
  //   prediction_id: 'pred_abc123',
  //   agent_id: 0,
  //   state: {
  //     aggressiveness: 8.2,
  //     consistency: 0.68,
  //     energy: -145.3
  //   }
  // }
});
```

#### Event Log

System events and milestones.

```javascript
socket.on('event_log', (data) => {
  // {
  //   prediction_id: 'pred_abc123',
  //   timestamp: '2025-11-12T10:35:22Z',
  //   level: 'info',
  //   message: 'New best energy: -189.2 kcal/mol',
  //   category: 'milestone'
  // }
});
```

#### Status Change

Prediction status changes.

```javascript
socket.on('status_change', (data) => {
  // {
  //   prediction_id: 'pred_abc123',
  //   old_status: 'running',
  //   new_status: 'completed'
  // }
});
```

#### Error Event

Errors during execution.

```javascript
socket.on('error_event', (data) => {
  // {
  //   prediction_id: 'pred_abc123',
  //   error: 'Energy calculation failed',
  //   details: { ... }
  // }
});
```

---

## Error Handling

All API endpoints follow consistent error response format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid protein sequence",
    "details": {
      "field": "sequence",
      "reason": "Contains invalid amino acid codes"
    }
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Request validation failed |
| `NOT_FOUND` | 404 | Resource not found |
| `CONFLICT` | 409 | Resource conflict |
| `RATE_LIMIT` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

---

## Rate Limiting

**Limits** (per IP):
- Create Prediction: 10 requests/minute
- List Operations: 60 requests/minute
- Other Endpoints: 30 requests/minute

**Headers**:
```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1699876543
```

When rate limit is exceeded, you'll receive a `429 Too Many Requests` response with a `Retry-After` header indicating when you can retry.

---

## Code Examples

### Python (requests)

```python
import requests

# Create prediction
response = requests.post('http://localhost:8000/api/predictions', json={
    'sequence': 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG',
    'iterations': 1000,
    'agents': 10,
    'enable_qcpp': True
})

prediction = response.json()
prediction_id = prediction['id']

# Get results
import time
while True:
    response = requests.get(f'http://localhost:8000/api/predictions/{prediction_id}')
    data = response.json()
    if data['status'] in ['completed', 'failed']:
        break
    time.sleep(5)

# Download structure
response = requests.get(f'http://localhost:8000/api/results/{prediction_id}/structure')
with open('structure.pdb', 'wb') as f:
    f.write(response.content)
```

### JavaScript (fetch)

```javascript
// Create prediction
const response = await fetch('http://localhost:8000/api/predictions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    sequence: 'MQIFVKT...',
    iterations: 1000,
    agents: 10,
    enable_qcpp: true
  })
});

const prediction = await response.json();

// WebSocket connection
import io from 'socket.io-client';
const socket = io('http://localhost:8000');

socket.emit('join_prediction', { prediction_id: prediction.id });

socket.on('progress_update', (data) => {
  console.log(`Progress: ${data.percentage}%`);
});
```

---

## Versioning

Current API version: **v1**

The API version is included in the base path: `/api/v1/...`

Future versions will be released as `/api/v2/`, `/api/v3/`, etc., with backward compatibility maintained for at least one major version.

---

## Support

For API issues or questions:
- Check [Troubleshooting Guide](TROUBLESHOOTING.md)
- Interactive docs: http://localhost:8000/docs
- GitHub Issues: <repository-url>/issues
