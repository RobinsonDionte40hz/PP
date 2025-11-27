# API Documentation - Protein Prediction Platform

This document describes the REST API endpoints for the Protein Prediction Platform.

## Table of Contents

- [Base URL](#base-url)
- [Authentication](#authentication)
- [Prediction Endpoints](#prediction-endpoints)
- [Campaign Endpoints](#campaign-endpoints)
- [Results Endpoints](#results-endpoints)
- [Work Session Endpoints](#work-session-endpoints)
- [WebSocket Events](#websocket-events)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)

## Base URL

**Development**: `http://localhost:8000`
**Production**: `https://your-domain.com`

All API endpoints are prefixed with `/api/v1` unless otherwise specified.

## Authentication

**Status**: ✅ Fully Implemented (v1.0.0)

The API uses JWT (JSON Web Tokens) for authentication with session management via Redis. All protected endpoints require a valid access token in the Authorization header.

### Authentication Flow

```
Registration → Login → Access Protected Resources → Token Refresh (Optional) → Logout
```

### Token Types

- **Access Token**: Short-lived (30 minutes), used for API requests
- **Refresh Token**: Long-lived (7 days), used to obtain new access tokens

### Using Authentication

Include the access token in the Authorization header for protected endpoints:

```http
Authorization: Bearer <access_token>
```

### Security Features

- ✅ Bcrypt password hashing (cost factor 12)
- ✅ JWT tokens with HMAC SHA-256 signing
- ✅ Single-session enforcement (new login terminates old session)
- ✅ Session storage in Redis with automatic expiration
- ✅ Rate limiting on auth endpoints
- ✅ CSRF protection
- ✅ Secure logging (no passwords/tokens in logs)

---

### Register User

Create a new user account.

**Endpoint**: `POST /api/auth/register`

**Rate Limit**: 5 requests per hour per IP

**Request Body**:
```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "SecurePass123!"
}
```

**Parameters**:
- `username` (string, required): 3-50 characters, alphanumeric with underscores and hyphens
- `email` (string, optional): Valid email address
- `password` (string, required): 8-72 characters, must contain uppercase, lowercase, digit, and special character

**Success Response** (201 Created):
```json
{
  "message": "User registered successfully",
  "user": {
    "key_id": "550e8400-e29b-41d4-a716-446655440000",
    "username": "john_doe",
    "email": "john@example.com",
    "created_at": "2025-11-22T10:30:00"
  }
}
```

**Error Responses**:
- `400 Bad Request`: Invalid input (see [Error Codes](#authentication-error-codes))
- `409 Conflict`: Username or email already exists
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

**Example**:
```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "email": "john@example.com",
    "password": "SecurePass123!"
  }'
```

---

### Login

Authenticate user and receive JWT tokens.

**Endpoint**: `POST /api/auth/login`

**Rate Limit**: 10 requests per 15 minutes per IP

**Request Body**:
```json
{
  "username": "john_doe",
  "password": "SecurePass123!"
}
```

**Parameters**:
- `username` (string, required): Username
- `password` (string, required): Password

**Success Response** (200 OK):
```json
{
  "message": "Login successful",
  "user": {
    "key_id": "550e8400-e29b-41d4-a716-446655440000",
    "username": "john_doe",
    "email": "john@example.com",
    "created_at": "2025-11-22T10:30:00"
  },
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Response Fields**:
- `access_token`: JWT for API requests (expires in 30 minutes)
- `refresh_token`: JWT for refreshing access token (expires in 7 days)
- `expires_in`: Access token expiration in seconds (1800 = 30 minutes)

**Error Responses**:
- `400 Bad Request`: Missing credentials
- `401 Unauthorized`: Invalid username or password
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

**Example**:
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "password": "SecurePass123!"
  }'
```

**Notes**:
- If user has an active session elsewhere, the old session will be terminated
- Creates a new session in Redis with 30-minute TTL
- Session automatically extends on activity

---

### Logout

Terminate the current session and invalidate tokens.

**Endpoint**: `POST /api/auth/logout`

**Authentication**: Required (Bearer token)

**Request Headers**:
```http
Authorization: Bearer <access_token>
```

**Success Response** (200 OK):
```json
{
  "message": "Logout successful"
}
```

**Error Responses**:
- `401 Unauthorized`: Invalid or missing token
- `500 Internal Server Error`: Server error

**Example**:
```bash
curl -X POST http://localhost:8000/api/auth/logout \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Notes**:
- Deletes session from Redis
- Removes user's active session mapping
- Client should discard tokens after logout

---

### Refresh Token

Obtain a new access token using a refresh token.

**Endpoint**: `POST /api/auth/refresh`

**Authentication**: Required (Refresh token)

**Request Body**:
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Success Response** (200 OK):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

**Error Responses**:
- `401 Unauthorized`: Invalid or expired refresh token
- `500 Internal Server Error`: Server error

**Example**:
```bash
curl -X POST http://localhost:8000/api/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }'
```

**Notes**:
- Use this endpoint to obtain a new access token before the current one expires
- Refresh tokens are valid for 7 days
- Updates session expiration in Redis

---

### Get Current User

Retrieve the authenticated user's profile.

**Endpoint**: `GET /api/auth/me`

**Authentication**: Required (Bearer token)

**Request Headers**:
```http
Authorization: Bearer <access_token>
```

**Success Response** (200 OK):
```json
{
  "key_id": "550e8400-e29b-41d4-a716-446655440000",
  "username": "john_doe",
  "email": "john@example.com",
  "created_at": "2025-11-22T10:30:00"
}
```

**Error Responses**:
- `401 Unauthorized`: Invalid or missing token
- `500 Internal Server Error`: Server error

**Example**:
```bash
curl -X GET http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

---

### Authentication Error Codes

Common error response format:
```json
{
  "detail": "Error message description"
}
```

#### Registration Errors

| Status Code | Detail Message | Reason |
|-------------|----------------|--------|
| 400 | Username must be 3-50 characters | Username too short or long |
| 400 | Username can only contain letters, numbers, underscores, and hyphens | Invalid characters in username |
| 400 | Password must be at least 8 characters | Password too short |
| 400 | Password must contain uppercase, lowercase, digit, and special character | Password doesn't meet complexity requirements |
| 400 | Invalid email format | Email format validation failed |
| 409 | Username already exists | Username is taken |
| 409 | Email already registered | Email is already in use |
| 429 | Too many registration attempts. Try again in X seconds | Rate limit exceeded (5/hour) |
| 500 | Internal server error | Server-side error |

#### Login Errors

| Status Code | Detail Message | Reason |
|-------------|----------------|--------|
| 400 | Username and password are required | Missing credentials |
| 401 | Invalid username or password | Authentication failed |
| 401 | Account locked due to too many failed attempts | Brute force protection triggered |
| 429 | Too many login attempts. Try again in X seconds | Rate limit exceeded (10/15min) |
| 500 | Internal server error | Server-side error |

#### Token Errors

| Status Code | Detail Message | Reason |
|-------------|----------------|--------|
| 401 | Invalid token | Token signature invalid or malformed |
| 401 | Token has expired | Access/refresh token expired |
| 401 | Session not found | Session was terminated or expired |
| 401 | Invalid token type | Wrong token type used (access vs refresh) |
| 500 | Internal server error | Server-side error |

#### Session Errors

| Status Code | Detail Message | Reason |
|-------------|----------------|--------|
| 401 | Session expired | Session TTL exceeded (30 minutes) |
| 401 | Session terminated | User logged out or session replaced |
| 401 | Invalid session | Session data corrupted or missing |

---

### Authentication Best Practices

#### Frontend Implementation

1. **Store Tokens Securely**:
   ```javascript
   // Store in memory or secure storage (not localStorage in production)
   localStorage.setItem('access_token', response.access_token);
   localStorage.setItem('refresh_token', response.refresh_token);
   ```

2. **Add Token to Requests**:
   ```javascript
   const config = {
     headers: {
       'Authorization': `Bearer ${access_token}`
     }
   };
   axios.get('/api/predictions', config);
   ```

3. **Handle Token Refresh**:
   ```javascript
   // Intercept 401 responses
   axios.interceptors.response.use(
     response => response,
     async error => {
       if (error.response?.status === 401) {
         const newToken = await refreshAccessToken();
         error.config.headers['Authorization'] = `Bearer ${newToken}`;
         return axios(error.config);
       }
       return Promise.reject(error);
     }
   );
   ```

4. **Auto-Refresh Before Expiration**:
   ```javascript
   // Refresh token 5 minutes before expiration
   const refreshTime = (expires_in - 300) * 1000;
   setTimeout(() => refreshAccessToken(), refreshTime);
   ```

#### Backend Integration

1. **Protect Routes**:
   ```python
   from app.middleware.auth import get_current_user
   
   @router.get("/protected")
   async def protected_route(current_user: User = Depends(get_current_user)):
       return {"user": current_user.username}
   ```

2. **Handle Errors**:
   ```python
   try:
       user = await auth_service.login(credentials)
   except HTTPException as e:
       # Handle specific errors
       if e.status_code == 401:
           return {"error": "Invalid credentials"}
       elif e.status_code == 429:
           return {"error": "Rate limited"}
   ```

---

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

## Work Session Endpoints

**Status**: ✅ Fully Implemented

Work sessions organize predictions into logical groups with isolated file storage, making it easy to manage and share related predictions.

### Key Features

- 📁 **Isolated Storage**: Each session has its own directory structure
- 🔐 **User Isolation**: Sessions are scoped to authenticated users
- 📦 **Easy Export**: Download entire session as ZIP archive
- 🔗 **Sharing**: Generate time-limited public share links
- 🧹 **Automatic Cleanup**: Expired sessions are automatically removed
- 🔄 **Activity Tracking**: Automatic timestamp updates on session access

### Configuration

Sessions are configured via environment variables (see `.env` or `backend/app/config.py`):

```bash
USER_DATA_DIR=./user_data                # Base directory for user data
SESSION_RETENTION_DAYS=90                # Inactive sessions deleted after 90 days
SHARE_LINK_MAX_HOURS=168                 # Share links expire after 7 days (168 hours)
CLEANUP_SCHEDULE_CRON="0 2 * * *"        # Daily cleanup at 2 AM
```

---

### List Work Sessions

Get paginated list of work sessions for authenticated user.

**Endpoint**: `GET /api/sessions`

**Authentication**: Required (JWT Bearer token)

**Rate Limit**: 30 requests/minute per user

**Query Parameters**:
- `page` (integer): Page number, 1-indexed (default: 1)
- `page_size` (integer): Items per page, max 100 (default: 20)

**Response** (200 OK):
```json
{
  "sessions": [
    {
      "id": "sess_abc123",
      "user_id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "Ubiquitin Study",
      "created_at": "2025-11-20T10:30:00Z",
      "last_active_at": "2025-11-25T15:45:00Z",
      "prediction_count": 5,
      "total_size_bytes": 15728640
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total": 42,
    "total_pages": 3
  }
}
```

**Response Fields**:
- `sessions`: Array of session objects
  - `id`: Unique session identifier
  - `user_id`: Owner's user ID
  - `name`: User-defined session name
  - `created_at`: Session creation timestamp
  - `last_active_at`: Last activity timestamp (auto-updated)
  - `prediction_count`: Number of predictions in session
  - `total_size_bytes`: Total storage size in bytes
- `pagination`: Pagination metadata

**Example**:
```bash
curl -X GET "http://localhost:8000/api/sessions?page=1&page_size=20" \
  -H "Authorization: Bearer <access_token>"
```

---

### Create Work Session

Create a new work session.

**Endpoint**: `POST /api/sessions`

**Authentication**: Required (JWT Bearer token)

**Rate Limit**: 30 requests/minute per user

**Request Body**:
```json
{
  "name": "Ubiquitin Study"
}
```

**Parameters**:
- `name` (string, required): Session name (1-200 characters)

**Response** (201 Created):
```json
{
  "id": "sess_abc123",
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Ubiquitin Study",
  "created_at": "2025-11-26T10:30:00Z",
  "last_active_at": "2025-11-26T10:30:00Z",
  "prediction_count": 0,
  "total_size_bytes": 0
}
```

**Error Responses**:
- `400 Bad Request`: Invalid session name
- `401 Unauthorized`: Missing or invalid authentication token
- `429 Too Many Requests`: Rate limit exceeded

**Example**:
```bash
curl -X POST http://localhost:8000/api/sessions \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "Ubiquitin Study"}'
```

---

### Get Work Session

Retrieve details of a specific work session.

**Endpoint**: `GET /api/sessions/{session_id}`

**Authentication**: Required (JWT Bearer token)

**Rate Limit**: 30 requests/minute per user

**Response** (200 OK):
```json
{
  "id": "sess_abc123",
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Ubiquitin Study",
  "created_at": "2025-11-20T10:30:00Z",
  "last_active_at": "2025-11-26T15:45:00Z",
  "prediction_count": 5,
  "total_size_bytes": 15728640
}
```

**Error Responses**:
- `401 Unauthorized`: Missing or invalid authentication token
- `403 Forbidden`: Session belongs to another user
- `404 Not Found`: Session does not exist

**Example**:
```bash
curl -X GET http://localhost:8000/api/sessions/sess_abc123 \
  -H "Authorization: Bearer <access_token>"
```

---

### Update Work Session

Update session name.

**Endpoint**: `PUT /api/sessions/{session_id}`

**Authentication**: Required (JWT Bearer token)

**Rate Limit**: 30 requests/minute per user

**Request Body**:
```json
{
  "name": "Updated Study Name"
}
```

**Parameters**:
- `name` (string, required): New session name (1-200 characters)

**Response** (200 OK):
```json
{
  "id": "sess_abc123",
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Updated Study Name",
  "created_at": "2025-11-20T10:30:00Z",
  "last_active_at": "2025-11-26T16:00:00Z",
  "prediction_count": 5,
  "total_size_bytes": 15728640
}
```

**Error Responses**:
- `400 Bad Request`: Invalid session name
- `401 Unauthorized`: Missing or invalid authentication token
- `403 Forbidden`: Session belongs to another user
- `404 Not Found`: Session does not exist

---

### Delete Work Session

Delete a work session including all predictions and files.

**Endpoint**: `DELETE /api/sessions/{session_id}`

**Authentication**: Required (JWT Bearer token)

**Rate Limit**: 30 requests/minute per user

**Response** (204 No Content):
- Empty response body on success

**Error Responses**:
- `401 Unauthorized`: Missing or invalid authentication token
- `403 Forbidden`: Session belongs to another user
- `404 Not Found`: Session does not exist

**Example**:
```bash
curl -X DELETE http://localhost:8000/api/sessions/sess_abc123 \
  -H "Authorization: Bearer <access_token>"
```

**Note**: This operation:
- Deletes the session database record
- Deletes all predictions in the session
- Removes the session directory and all files
- Cannot be undone

---

### List Session Predictions

Get paginated list of predictions in a work session.

**Endpoint**: `GET /api/sessions/{session_id}/predictions`

**Authentication**: Required (JWT Bearer token)

**Rate Limit**: 30 requests/minute per user

**Query Parameters**:
- `page` (integer): Page number, 1-indexed (default: 1)
- `page_size` (integer): Items per page, max 100 (default: 20)

**Response** (200 OK):
```json
{
  "predictions": [
    {
      "id": "pred_abc123",
      "session_id": "sess_abc123",
      "sequence": "MQIFVKT...",
      "status": "completed",
      "config": {
        "iterations": 1000,
        "agents": 10
      },
      "created_at": "2025-11-26T10:35:00Z",
      "completed_at": "2025-11-26T10:50:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "page_size": 20,
    "total": 5,
    "total_pages": 1
  }
}
```

**Error Responses**:
- `401 Unauthorized`: Missing or invalid authentication token
- `403 Forbidden`: Session belongs to another user
- `404 Not Found`: Session does not exist

---

### Create Prediction in Session

Create a new prediction within a work session.

**Endpoint**: `POST /api/sessions/{session_id}/predictions`

**Authentication**: Required (JWT Bearer token)

**Rate Limit**: 10 requests/minute per user

**Request Body**:
```json
{
  "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
  "iterations": 1000,
  "agents": 10,
  "enable_qcpp": true,
  "enable_mediators": true,
  "enable_geometric_targeting": true,
  "enable_refinement": true
}
```

**Parameters**: Same as [Create Prediction](#create-prediction), but automatically linked to the session.

**Response** (201 Created):
```json
{
  "id": "pred_abc123",
  "session_id": "sess_abc123",
  "status": "queued",
  "sequence": "MQIFVKT...",
  "config": {
    "iterations": 1000,
    "agents": 10
  },
  "created_at": "2025-11-26T16:30:00Z"
}
```

**Error Responses**:
- `400 Bad Request`: Invalid sequence or parameters
- `401 Unauthorized`: Missing or invalid authentication token
- `403 Forbidden`: Session belongs to another user
- `404 Not Found`: Session does not exist
- `429 Too Many Requests`: Rate limit exceeded

**Example**:
```bash
curl -X POST http://localhost:8000/api/sessions/sess_abc123/predictions \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "MQIFVKTLTGKTIT...",
    "iterations": 1000,
    "agents": 10
  }'
```

**Note**: This automatically:
- Links the prediction to the session via `session_id`
- Updates the session's `last_active_at` timestamp
- Stores prediction files in `user_data/{user_id}/sessions/{session_id}/{prediction_id}/`

---

### Download Session

Download entire work session as ZIP archive.

**Endpoint**: `GET /api/sessions/{session_id}/download`

**Authentication**: Required (JWT Bearer token)

**Rate Limit**: 10 requests/minute per user

**Response** (200 OK):
- `Content-Type: application/zip`
- `Content-Disposition: attachment; filename="session_{session_id}.zip"`
- Binary ZIP file containing:
  - `metadata.json`: Session metadata
  - `{prediction_id}/`: Directory for each prediction
    - `results.json`: Prediction results
    - `trajectory.json`: Conformational trajectory
    - `structure.pdb`: Final protein structure
    - `visualization.png`: Structure visualization

**Error Responses**:
- `401 Unauthorized`: Missing or invalid authentication token
- `403 Forbidden`: Session belongs to another user
- `404 Not Found`: Session does not exist
- `500 Internal Server Error`: ZIP creation failed

**Example**:
```bash
curl -X GET http://localhost:8000/api/sessions/sess_abc123/download \
  -H "Authorization: Bearer <access_token>" \
  -o session.zip
```

**ZIP Structure**:
```
session_sess_abc123.zip
├── metadata.json
├── pred_abc123/
│   ├── results.json
│   ├── trajectory.json
│   ├── structure.pdb
│   └── visualization.png
└── pred_def456/
    ├── results.json
    ├── trajectory.json
    ├── structure.pdb
    └── visualization.png
```

---

### Create Share Link

Generate a time-limited public share link for a session.

**Endpoint**: `POST /api/sessions/{session_id}/share`

**Authentication**: Required (JWT Bearer token)

**Rate Limit**: 30 requests/minute per user

**Request Body**:
```json
{
  "expires_in_hours": 168
}
```

**Parameters**:
- `expires_in_hours` (integer, optional): Hours until expiration (default: 168, max: 168)

**Response** (201 Created):
```json
{
  "share_id": "sh_xyz789",
  "session_id": "sess_abc123",
  "share_url": "http://localhost:8000/api/shared/sh_xyz789",
  "created_at": "2025-11-26T16:30:00Z",
  "expires_at": "2025-12-03T16:30:00Z",
  "access_count": 0
}
```

**Response Fields**:
- `share_id`: Unique share identifier
- `session_id`: Associated session ID
- `share_url`: Full URL for accessing shared session
- `created_at`: Share link creation timestamp
- `expires_at`: Share link expiration timestamp
- `access_count`: Number of times link has been accessed

**Error Responses**:
- `400 Bad Request`: Invalid expiration time
- `401 Unauthorized`: Missing or invalid authentication token
- `403 Forbidden`: Session belongs to another user
- `404 Not Found`: Session does not exist

**Example**:
```bash
curl -X POST http://localhost:8000/api/sessions/sess_abc123/share \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"expires_in_hours": 168}'
```

---

### Access Shared Session

Access a shared session via public share link (no authentication required).

**Endpoint**: `GET /api/shared/{share_id}`

**Authentication**: Not required (public endpoint)

**Rate Limit**: 30 requests/minute per IP

**Response** (200 OK):
```json
{
  "session": {
    "id": "sess_abc123",
    "name": "Ubiquitin Study",
    "created_at": "2025-11-20T10:30:00Z",
    "prediction_count": 5
  },
  "predictions": [
    {
      "id": "pred_abc123",
      "sequence": "MQIFVKT...",
      "status": "completed",
      "created_at": "2025-11-26T10:35:00Z",
      "results_summary": {
        "final_energy": -189.2,
        "final_rmsd": 7.2
      }
    }
  ],
  "share_info": {
    "expires_at": "2025-12-03T16:30:00Z",
    "access_count": 15
  }
}
```

**Response Fields**:
- `session`: Read-only session information (excludes user_id)
- `predictions`: Array of predictions in the session (summary only)
- `share_info`: Share link metadata

**Error Responses**:
- `404 Not Found`: Share link does not exist or has expired
- `429 Too Many Requests`: Rate limit exceeded

**Example**:
```bash
curl -X GET http://localhost:8000/api/shared/sh_xyz789
```

**Note**: 
- This is a **read-only** endpoint
- No authentication required
- Increments `access_count` on each access
- Returns 404 if share link has expired

---

### File Storage Structure

Work sessions organize files with user and session isolation:

```
user_data/
└── {user_id}/
    └── sessions/
        └── {session_id}/
            ├── {prediction_id}/
            │   ├── results.json
            │   ├── trajectory.json
            │   ├── structure.pdb
            │   └── visualization.png
            └── {prediction_id}/
                ├── results.json
                ├── trajectory.json
                ├── structure.pdb
                └── visualization.png
```

**Isolation Guarantees**:
- Each user has isolated directory: `user_data/{user_id}/`
- Each session has isolated subdirectory: `sessions/{session_id}/`
- Each prediction has isolated subdirectory: `{prediction_id}/`
- All file operations validate ownership

---

### Automatic Cleanup

The system automatically maintains storage by removing expired sessions.

**Cleanup Process**:
1. Runs daily at 2 AM (configurable via `CLEANUP_SCHEDULE_CRON`)
2. Identifies sessions with `last_active_at` older than `SESSION_RETENTION_DAYS` (default: 90 days)
3. Deletes database records (WorkSession, SharedExport, Prediction)
4. Removes file system directories
5. Logs all cleanup operations

**Configuration**:
```bash
SESSION_RETENTION_DAYS=90                # Retention period
CLEANUP_SCHEDULE_CRON="0 2 * * *"        # Cleanup schedule (cron format)
```

**Manual Cleanup**:
Cleanup can be triggered manually via the cleanup service:
```python
from app.services.session_cleanup_service import get_cleanup_service

cleanup_service = get_cleanup_service()
stats = cleanup_service.delete_expired_sessions(retention_days=90)
print(f"Deleted {stats['sessions_deleted']} sessions")
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
