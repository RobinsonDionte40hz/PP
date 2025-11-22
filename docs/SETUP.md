# Setup Guide - Protein Prediction Platform Web Interface

This guide will help you set up and run the Protein Prediction Platform web interface.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start (Docker)](#quick-start-docker)
- [Development Setup](#development-setup)
- [Environment Configuration](#environment-configuration)
- [Running the Application](#running-the-application)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

- **Docker**: 24.0+ and Docker Compose 2.20+
- **Node.js**: 18+ (for frontend development)
- **Python**: 3.8-3.12 (for backend development)
- **Git**: For cloning the repository

### System Requirements

- **OS**: Windows 10/11, macOS 10.15+, or Linux
- **RAM**: 8GB minimum, 16GB recommended
- **Disk Space**: 10GB free space
- **CPU**: Multi-core processor recommended for parallel predictions

## Quick Start (Docker)

The fastest way to get started is using Docker Compose, which will set up all services automatically.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd PP
```

### 2. Configure Environment

Copy the example environment file and edit it:

```bash
cp .env.example .env
```

Edit `.env` with your preferred settings. For a quick start, the defaults should work.

### 3. Build and Start Services

```bash
docker-compose up --build
```

This will:
- Build frontend (React + Vite)
- Build backend (FastAPI)
- Build worker (Celery)
- Start Redis
- Start PostgreSQL (optional)

### 4. Access the Application

Once all services are running:

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Redis**: localhost:6379
- **PostgreSQL**: localhost:5432

### 5. Verify Installation

Navigate to http://localhost:3000 and you should see the dashboard.

To verify backend:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "services": {
    "redis": "connected",
    "database": "connected",
    "pp_system": "ready"
  }
}
```

## Development Setup

For active development, you may want to run services outside Docker for faster iteration.

### Backend Development

#### 1. Create Virtual Environment

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 3. Set Environment Variables

Create `backend/.env`:
```bash
DATABASE_URL=postgresql://user:password@localhost:5432/pp_db
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=dev-secret-key
APP_ENV=development
```

#### 4. Start Redis (if not using Docker)

Windows (using Docker):
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

macOS (using Homebrew):
```bash
brew install redis
brew services start redis
```

Linux (Ubuntu/Debian):
```bash
sudo apt install redis-server
sudo systemctl start redis-server
```

#### 5. Run Database Migrations (if using PostgreSQL)

```bash
# Install Alembic if using migrations
pip install alembic

# Run migrations
alembic upgrade head
```

#### 6. Start Backend Server

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

#### 7. Start Celery Worker (separate terminal)

```bash
cd backend
celery -A app.celery_app:celery_app worker --loglevel=info
```

Windows users may need to add the `--pool=solo` flag:
```bash
celery -A app.celery_app:celery_app worker --loglevel=info --pool=solo
```

### Frontend Development

#### 1. Install Dependencies

```bash
cd frontend
npm install
```

#### 2. Configure Environment

Create `frontend/.env`:
```bash
VITE_API_URL=http://localhost:8000
VITE_WS_URL=http://localhost:8000
```

#### 3. Start Development Server

```bash
npm run dev
```

The frontend will be available at http://localhost:5173 (Vite's default port).

#### 4. Build for Production

```bash
npm run build
```

Production files will be in `frontend/dist/`.

## Environment Configuration

### Backend Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection string | - | No |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` | Yes |
| `SECRET_KEY` | Secret key for sessions/JWT | - | **Yes** |
| `APP_ENV` | Environment (development/production) | `development` | No |
| `PP_RESULTS_DIR` | Directory for PP results | `./results` | No |
| `PP_CHECKPOINTS_DIR` | Directory for checkpoints | `./checkpoints` | No |
| `PP_PDB_CACHE_DIR` | Directory for PDB cache | `./pdb_cache` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `CORS_ORIGINS` | Allowed CORS origins | `["http://localhost:3000"]` | No |

### Authentication Configuration (NEW - v1.0.0)

The platform includes JWT-based authentication with Redis session management. Authentication is **required** for all protected endpoints.

#### Authentication Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `SECRET_KEY` | JWT signing key (HMAC SHA-256) | - | **Yes** |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | Access token lifetime | `30` | No |
| `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | Refresh token lifetime | `7` | No |
| `JWT_ALGORITHM` | JWT signing algorithm | `HS256` | No |
| `SESSION_TTL_SECONDS` | Redis session TTL | `1800` (30min) | No |
| `ENABLE_CSRF` | Enable CSRF protection | `true` | No |
| `ENABLE_HSTS` | Enable HSTS headers | `false` (dev) | No |

#### Generating a Secure Secret Key

**CRITICAL**: Never use default secret keys in production!

**Generate a secure key**:

Python method:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

OpenSSL method:
```bash
openssl rand -base64 32
```

**Add to `.env`**:
```bash
SECRET_KEY=your-secure-random-key-here
```

#### Redis Configuration for Authentication

Redis is **required** for session management:

**Docker (recommended)**:
```bash
# Already included in docker-compose.yml
docker-compose up redis
```

**Windows**:
```bash
# Using Docker
docker run -d -p 6379:6379 --name redis-auth redis:7-alpine
```

**macOS**:
```bash
brew install redis
brew services start redis
```

**Linux**:
```bash
sudo apt-get install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

**Test Redis connection**:
```bash
redis-cli ping
# Should respond: PONG
```

#### Database Migration for Users Table

Run the authentication database migration to create the users table:

```bash
cd backend
python -m app.migrations.create_users_table
```

This creates:
- `users` table with username, email, password_hash
- UUID-based key_id for each user
- Indexes on username and email for performance
- Timestamps for created_at and last_login

**Verify table creation**:
```bash
# If using SQLite (default)
sqlite3 backend/app/pp_database.db ".schema users"

# If using PostgreSQL
psql -d pp_db -c "\d users"
```

#### First User Registration

After setup, register your first user:

**Via API**:
```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "email": "admin@example.com",
    "password": "SecurePass123!"
  }'
```

**Via Frontend**:
1. Navigate to http://localhost:3000/register
2. Fill in the registration form
3. Submit to create your account
4. Automatically redirected to login

#### Security Best Practices

**For Development**:
```bash
SECRET_KEY=dev-secret-key-change-in-production
APP_ENV=development
ENABLE_CSRF=true
ENABLE_HSTS=false
CORS_ORIGINS=["http://localhost:3000"]
```

**For Production**:
```bash
SECRET_KEY=<secure-random-32-byte-key>
APP_ENV=production
ENABLE_CSRF=true
ENABLE_HSTS=true
CORS_ORIGINS=["https://yourdomain.com"]
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15  # Shorter tokens in production
SESSION_TTL_SECONDS=900  # 15 minutes idle timeout
```

**Additional Security Recommendations**:
- ✅ Use HTTPS in production (required for HSTS)
- ✅ Set strong password requirements (enforced by default)
- ✅ Monitor rate limiting logs for abuse
- ✅ Rotate secret keys periodically
- ✅ Use separate Redis database for sessions (REDIS_URL with `/1` or `/2`)
- ✅ Enable Redis persistence for session durability
- ✅ Set up Redis password authentication in production

#### Authentication Testing

**Test authentication flow**:
```bash
# 1. Register user
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"TestPass123!"}'

# 2. Login
TOKEN=$(curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"TestPass123!"}' \
  | jq -r '.access_token')

# 3. Access protected endpoint
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/auth/me

# 4. Logout
curl -X POST http://localhost:8000/api/auth/logout \
  -H "Authorization: Bearer $TOKEN"
```

**Verify session in Redis**:
```bash
redis-cli
> KEYS session:*
> GET session:<jti-from-token>
```

#### Troubleshooting Authentication

| Issue | Cause | Solution |
|-------|-------|----------|
| 500 error on login | Redis not running | Start Redis service |
| Invalid signature | Wrong SECRET_KEY | Check .env file matches |
| Session not found | Redis restarted | Login again to create new session |
| CSRF token mismatch | Missing CSRF header | Include X-CSRF-Token header |
| Rate limited | Too many attempts | Wait for retry-after period |

#### Authentication Documentation

For detailed API documentation, see:
- **API Reference**: `docs/API.md#authentication`
- **Authentication Flows**: `docs/AUTHENTICATION_FLOWS.md`
- **Error Handling**: `docs/ERROR_HANDLING.md`
- **Security Guide**: `backend/SECURITY.md`

### Frontend Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `VITE_API_URL` | Backend API URL | `http://localhost:8000` | Yes |
| `VITE_WS_URL` | WebSocket server URL | `http://localhost:8000` | Yes |

### Docker Compose Environment Variables

See `.env.example` for the full list. Key variables:

- `POSTGRES_USER`: PostgreSQL username
- `POSTGRES_PASSWORD`: PostgreSQL password
- `POSTGRES_DB`: Database name
- `REDIS_PORT`: Redis port mapping
- `BACKEND_PORT`: Backend API port mapping
- `FRONTEND_PORT`: Frontend port mapping

## Running the Application

### Using Docker (Recommended)

**Start all services**:
```bash
docker-compose up
```

**Start in detached mode**:
```bash
docker-compose up -d
```

**Stop services**:
```bash
docker-compose down
```

**Rebuild after code changes**:
```bash
docker-compose up --build
```

**View logs**:
```bash
docker-compose logs -f

# Specific service
docker-compose logs -f backend
```

### Using Development Setup

1. **Start Redis** (if not using Docker)
2. **Start Backend**: `cd backend && uvicorn app.main:app --reload`
3. **Start Celery**: `cd backend && celery -A app.celery_app:celery_app worker --loglevel=info`
4. **Start Frontend**: `cd frontend && npm run dev`

## Troubleshooting

### Port Conflicts

If ports 3000, 8000, 6379, or 5432 are already in use:

1. **Check what's using the port** (Windows):
   ```bash
   netstat -ano | findstr :8000
   ```

2. **Change ports in docker-compose.yml** or environment variables

### Docker Build Failures

1. **Clear Docker cache**:
   ```bash
   docker system prune -a
   ```

2. **Rebuild without cache**:
   ```bash
   docker-compose build --no-cache
   ```

### Permission Issues (Linux/macOS)

If you encounter permission errors with volumes:

```bash
sudo chown -R $USER:$USER ./checkpoints ./visualization_output ./pdb_cache
```

### Redis Connection Errors

1. **Check Redis is running**:
   ```bash
   redis-cli ping
   ```
   Should return `PONG`

2. **Check connection string** in `.env`

3. **For Docker**, ensure Redis container is running:
   ```bash
   docker-compose ps
   ```

### Database Connection Errors

1. **Check PostgreSQL is running**:
   ```bash
   docker-compose ps postgres
   ```

2. **Verify connection string** in `.env`

3. **Check database exists**:
   ```bash
   docker-compose exec postgres psql -U user -d pp_db -c "\l"
   ```

### Frontend Build Errors

1. **Clear node_modules and reinstall**:
   ```bash
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

2. **Check Node version**:
   ```bash
   node --version  # Should be 18+
   ```

### Backend Import Errors

1. **Ensure virtual environment is activated**
2. **Reinstall dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Celery Worker Not Starting (Windows)

Windows users may need the `--pool=solo` flag:
```bash
celery -A app.celery_app:celery_app worker --loglevel=info --pool=solo
```

### WebSocket Connection Issues

1. **Check Socket.IO version compatibility** between frontend and backend
2. **Verify CORS settings** in backend
3. **Check firewall rules** for WebSocket connections

### PP System Integration Issues

1. **Verify PP system files** exist:
   - `test_protein.py`
   - `systematic_protein_testing.py`
   - `ubf_protein/` directory

2. **Check Python version** (3.8-3.12 recommended)

3. **Install PP dependencies**:
   ```bash
   pip install -r requirements_qcpp.txt
   pip install -r ubf_protein/requirements.txt
   ```

## Next Steps

- [User Guide](USER_GUIDE.md) - Learn how to use the interface
- [Developer Guide](DEVELOPER_GUIDE.md) - Learn about the codebase
- [API Documentation](API.md) - Explore the REST API
- [Troubleshooting Guide](TROUBLESHOOTING.md) - More detailed troubleshooting

## Getting Help

If you encounter issues not covered in this guide:

1. Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
2. Review logs: `docker-compose logs` or check browser console
3. Check GitHub Issues for known problems
4. Create a new issue with detailed information about your setup and the error
