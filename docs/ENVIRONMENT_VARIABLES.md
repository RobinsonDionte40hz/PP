# Environment Variables Reference

This document describes all environment variables used by the Protein Prediction Platform.

## Table of Contents

- [Backend Variables](#backend-variables)
- [Frontend Variables](#frontend-variables)
- [Docker Variables](#docker-variables)
- [Example Configurations](#example-configurations)

## Backend Variables

### Core Settings

#### `APP_ENV`
- **Description**: Application environment
- **Type**: string
- **Values**: `development`, `production`, `testing`
- **Default**: `development`
- **Required**: No
- **Example**: `APP_ENV=production`

#### `SECRET_KEY`
- **Description**: Secret key for cryptographic operations (sessions, JWT, etc.)
- **Type**: string
- **Default**: None
- **Required**: Yes
- **Example**: `SECRET_KEY=your-secret-key-min-32-chars`
- **Security**: Use a strong random string (min 32 characters). Generate with:
  ```bash
  python -c "import secrets; print(secrets.token_urlsafe(32))"
  ```

#### `LOG_LEVEL`
- **Description**: Logging level
- **Type**: string
- **Values**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Default**: `INFO`
- **Required**: No
- **Example**: `LOG_LEVEL=DEBUG`

### Database

#### `DATABASE_URL`
- **Description**: PostgreSQL connection string (optional - in-memory fallback)
- **Type**: string
- **Format**: `postgresql://user:password@host:port/database`
- **Default**: None (uses in-memory storage)
- **Required**: No
- **Example**: `DATABASE_URL=postgresql://ppuser:password@localhost:5432/pp_db`

### Redis

#### `REDIS_URL`
- **Description**: Redis connection string
- **Type**: string
- **Format**: `redis://host:port/db`
- **Default**: `redis://localhost:6379/0`
- **Required**: Yes
- **Example**: `REDIS_URL=redis://localhost:6379/0`

#### `REDIS_PASSWORD`
- **Description**: Redis password (if authentication enabled)
- **Type**: string
- **Default**: None
- **Required**: No
- **Example**: `REDIS_PASSWORD=your-redis-password`

### Celery

#### `CELERY_BROKER_URL`
- **Description**: Celery broker URL (usually same as REDIS_URL)
- **Type**: string
- **Default**: Value of `REDIS_URL`
- **Required**: No
- **Example**: `CELERY_BROKER_URL=redis://localhost:6379/0`

#### `CELERY_RESULT_BACKEND`
- **Description**: Celery result backend URL
- **Type**: string
- **Default**: Value of `REDIS_URL`
- **Required**: No
- **Example**: `CELERY_RESULT_BACKEND=redis://localhost:6379/0`

#### `CELERY_TASK_TIME_LIMIT`
- **Description**: Maximum time (seconds) a task can run
- **Type**: integer
- **Default**: `7200` (2 hours)
- **Required**: No
- **Example**: `CELERY_TASK_TIME_LIMIT=10800`

### CORS

#### `CORS_ORIGINS`
- **Description**: Allowed CORS origins (comma-separated)
- **Type**: string (comma-separated list)
- **Default**: `http://localhost:3000,http://localhost:5173`
- **Required**: No
- **Example**: `CORS_ORIGINS=http://localhost:3000,https://yourdomain.com`

#### `CORS_ALLOW_CREDENTIALS`
- **Description**: Allow credentials in CORS requests
- **Type**: boolean
- **Values**: `true`, `false`
- **Default**: `true`
- **Required**: No
- **Example**: `CORS_ALLOW_CREDENTIALS=true`

### PP System Integration

#### `PP_RESULTS_DIR`
- **Description**: Directory for PP system results
- **Type**: string (path)
- **Default**: `./results`
- **Required**: No
- **Example**: `PP_RESULTS_DIR=/var/pp/results`

#### `PP_CHECKPOINTS_DIR`
- **Description**: Directory for prediction checkpoints
- **Type**: string (path)
- **Default**: `./checkpoints`
- **Required**: No
- **Example**: `PP_CHECKPOINTS_DIR=/var/pp/checkpoints`

#### `PP_PDB_CACHE_DIR`
- **Description**: Directory for cached PDB files
- **Type**: string (path)
- **Default**: `./pdb_cache`
- **Required**: No
- **Example**: `PP_PDB_CACHE_DIR=/var/pp/pdb_cache`

#### `PP_MAX_CONCURRENT_PREDICTIONS`
- **Description**: Maximum number of concurrent predictions
- **Type**: integer
- **Default**: `5`
- **Required**: No
- **Example**: `PP_MAX_CONCURRENT_PREDICTIONS=10`

#### `PP_AUTO_CLEANUP_DAYS`
- **Description**: Days after which old results are auto-deleted (0=disabled)
- **Type**: integer
- **Default**: `30`
- **Required**: No
- **Example**: `PP_AUTO_CLEANUP_DAYS=60`

### Performance

#### `MAX_UPLOAD_SIZE_MB`
- **Description**: Maximum file upload size in MB
- **Type**: integer
- **Default**: `10`
- **Required**: No
- **Example**: `MAX_UPLOAD_SIZE_MB=50`

#### `CACHE_TTL_SECONDS`
- **Description**: Default cache TTL in seconds
- **Type**: integer
- **Default**: `300` (5 minutes)
- **Required**: No
- **Example**: `CACHE_TTL_SECONDS=600`

### Rate Limiting

#### `RATE_LIMIT_ENABLED`
- **Description**: Enable rate limiting
- **Type**: boolean
- **Values**: `true`, `false`
- **Default**: `true`
- **Required**: No
- **Example**: `RATE_LIMIT_ENABLED=false`

#### `RATE_LIMIT_PER_MINUTE`
- **Description**: Max requests per minute per IP
- **Type**: integer
- **Default**: `60`
- **Required**: No
- **Example**: `RATE_LIMIT_PER_MINUTE=120`

### WebSocket

#### `WS_HEARTBEAT_INTERVAL`
- **Description**: WebSocket heartbeat interval (seconds)
- **Type**: integer
- **Default**: `25`
- **Required**: No
- **Example**: `WS_HEARTBEAT_INTERVAL=30`

#### `WS_MAX_CONNECTIONS`
- **Description**: Maximum concurrent WebSocket connections
- **Type**: integer
- **Default**: `100`
- **Required**: No
- **Example**: `WS_MAX_CONNECTIONS=200`

### Monitoring & Logging

#### `SENTRY_DSN`
- **Description**: Sentry DSN for error tracking (optional)
- **Type**: string (URL)
- **Default**: None
- **Required**: No
- **Example**: `SENTRY_DSN=https://key@sentry.io/project`

#### `LOG_FILE`
- **Description**: Log file path
- **Type**: string (path)
- **Default**: None (logs to stdout)
- **Required**: No
- **Example**: `LOG_FILE=/var/log/pp/app.log`

#### `LOG_JSON`
- **Description**: Output logs in JSON format
- **Type**: boolean
- **Values**: `true`, `false`
- **Default**: `false`
- **Required**: No
- **Example**: `LOG_JSON=true`

---

## Frontend Variables

All frontend variables must be prefixed with `VITE_` to be available in the browser.

### API Configuration

#### `VITE_API_URL`
- **Description**: Backend API base URL
- **Type**: string (URL)
- **Default**: `http://localhost:8000`
- **Required**: Yes
- **Example**: `VITE_API_URL=https://api.yourdomain.com`

#### `VITE_WS_URL`
- **Description**: WebSocket server URL
- **Type**: string (URL)
- **Default**: `http://localhost:8000`
- **Required**: Yes
- **Example**: `VITE_WS_URL=https://api.yourdomain.com`

### Feature Flags

#### `VITE_ENABLE_3D_VIEWER`
- **Description**: Enable 3D structure viewer
- **Type**: boolean
- **Values**: `true`, `false`
- **Default**: `true`
- **Required**: No
- **Example**: `VITE_ENABLE_3D_VIEWER=false`

#### `VITE_ENABLE_CAMPAIGNS`
- **Description**: Enable campaign management
- **Type**: boolean
- **Values**: `true`, `false`
- **Default**: `true`
- **Required**: No
- **Example**: `VITE_ENABLE_CAMPAIGNS=false`

#### `VITE_ENABLE_ANALYTICS`
- **Description**: Enable analytics tracking
- **Type**: boolean
- **Values**: `true`, `false`
- **Default**: `false`
- **Required**: No
- **Example**: `VITE_ENABLE_ANALYTICS=true`

### UI Configuration

#### `VITE_DEFAULT_THEME`
- **Description**: Default UI theme
- **Type**: string
- **Values**: `light`, `dark`
- **Default**: `light`
- **Required**: No
- **Example**: `VITE_DEFAULT_THEME=dark`

#### `VITE_ITEMS_PER_PAGE`
- **Description**: Default items per page in lists
- **Type**: integer
- **Default**: `10`
- **Required**: No
- **Example**: `VITE_ITEMS_PER_PAGE=20`

### External Services

#### `VITE_GA_TRACKING_ID`
- **Description**: Google Analytics tracking ID (optional)
- **Type**: string
- **Default**: None
- **Required**: No
- **Example**: `VITE_GA_TRACKING_ID=G-XXXXXXXXXX`

---

## Docker Variables

### PostgreSQL

#### `POSTGRES_USER`
- **Description**: PostgreSQL username
- **Type**: string
- **Default**: `ppuser`
- **Required**: Yes (for Docker)
- **Example**: `POSTGRES_USER=ppuser`

#### `POSTGRES_PASSWORD`
- **Description**: PostgreSQL password
- **Type**: string
- **Default**: `pppassword`
- **Required**: Yes (for Docker)
- **Example**: `POSTGRES_PASSWORD=secure-password`
- **Security**: Use strong password in production

#### `POSTGRES_DB`
- **Description**: PostgreSQL database name
- **Type**: string
- **Default**: `pp_db`
- **Required**: Yes (for Docker)
- **Example**: `POSTGRES_DB=pp_production`

### Port Mappings

#### `FRONTEND_PORT`
- **Description**: Frontend external port
- **Type**: integer
- **Default**: `3000`
- **Required**: No
- **Example**: `FRONTEND_PORT=8080`

#### `BACKEND_PORT`
- **Description**: Backend external port
- **Type**: integer
- **Default**: `8000`
- **Required**: No
- **Example**: `BACKEND_PORT=8001`

#### `REDIS_PORT`
- **Description**: Redis external port
- **Type**: integer
- **Default**: `6379`
- **Required**: No
- **Example**: `REDIS_PORT=6380`

#### `POSTGRES_PORT`
- **Description**: PostgreSQL external port
- **Type**: integer
- **Default**: `5432`
- **Required**: No
- **Example**: `POSTGRES_PORT=5433`

---

## Example Configurations

### Development

```bash
# Backend (.env)
APP_ENV=development
SECRET_KEY=dev-secret-key-for-testing-only
LOG_LEVEL=DEBUG

# Database (optional in dev)
DATABASE_URL=postgresql://ppuser:pppassword@localhost:5432/pp_db

# Redis
REDIS_URL=redis://localhost:6379/0

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# PP System
PP_RESULTS_DIR=./results
PP_CHECKPOINTS_DIR=./checkpoints
PP_PDB_CACHE_DIR=./pdb_cache
PP_MAX_CONCURRENT_PREDICTIONS=3

# Frontend (.env)
VITE_API_URL=http://localhost:8000
VITE_WS_URL=http://localhost:8000
VITE_DEFAULT_THEME=light
VITE_ENABLE_3D_VIEWER=true
VITE_ENABLE_CAMPAIGNS=true
```

### Production

```bash
# Backend (.env)
APP_ENV=production
SECRET_KEY=<generate-strong-random-key>
LOG_LEVEL=INFO
LOG_JSON=true
LOG_FILE=/var/log/pp/app.log

# Database
DATABASE_URL=postgresql://ppuser:<strong-password>@postgres:5432/pp_production

# Redis
REDIS_URL=redis://redis:6379/0
REDIS_PASSWORD=<redis-password>

# Celery
CELERY_TASK_TIME_LIMIT=10800

# CORS
CORS_ORIGINS=https://yourdomain.com
CORS_ALLOW_CREDENTIALS=true

# PP System
PP_RESULTS_DIR=/var/pp/results
PP_CHECKPOINTS_DIR=/var/pp/checkpoints
PP_PDB_CACHE_DIR=/var/pp/pdb_cache
PP_MAX_CONCURRENT_PREDICTIONS=10
PP_AUTO_CLEANUP_DAYS=60

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=30

# Monitoring
SENTRY_DSN=https://<key>@sentry.io/<project>

# Frontend (.env)
VITE_API_URL=https://api.yourdomain.com
VITE_WS_URL=https://api.yourdomain.com
VITE_DEFAULT_THEME=light
VITE_ENABLE_3D_VIEWER=true
VITE_ENABLE_CAMPAIGNS=true
VITE_ENABLE_ANALYTICS=true
VITE_GA_TRACKING_ID=G-XXXXXXXXXX
```

### Testing

```bash
# Backend (.env.test)
APP_ENV=testing
SECRET_KEY=test-secret-key
LOG_LEVEL=DEBUG

# Use in-memory database
DATABASE_URL=

# Redis
REDIS_URL=redis://localhost:6379/1

# Disable rate limiting for tests
RATE_LIMIT_ENABLED=false

# PP System
PP_RESULTS_DIR=./test_results
PP_CHECKPOINTS_DIR=./test_checkpoints
PP_PDB_CACHE_DIR=./test_pdb_cache
PP_MAX_CONCURRENT_PREDICTIONS=2

# Frontend (.env.test)
VITE_API_URL=http://localhost:8000
VITE_WS_URL=http://localhost:8000
VITE_ENABLE_3D_VIEWER=false
VITE_ENABLE_CAMPAIGNS=false
VITE_ENABLE_ANALYTICS=false
```

### Docker Compose

```bash
# .env (for docker-compose.yml)
# PostgreSQL
POSTGRES_USER=ppuser
POSTGRES_PASSWORD=secure-production-password
POSTGRES_DB=pp_db

# Port Mappings
FRONTEND_PORT=3000
BACKEND_PORT=8000
REDIS_PORT=6379
POSTGRES_PORT=5432

# Application
APP_ENV=production
SECRET_KEY=<generate-strong-random-key>

# PP System
PP_MAX_CONCURRENT_PREDICTIONS=10
PP_AUTO_CLEANUP_DAYS=60
```

---

## Security Best Practices

### Production Secrets

1. **Never commit `.env` files** to version control
   - Add `.env` to `.gitignore`
   - Use `.env.example` as template

2. **Use strong random keys**:
   ```bash
   # Generate SECRET_KEY
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   
   # Generate database password
   python -c "import secrets; print(secrets.token_urlsafe(24))"
   ```

3. **Rotate secrets regularly**
   - Change SECRET_KEY every 90 days
   - Change database passwords every 180 days

4. **Use environment-specific files**:
   - `.env.development`
   - `.env.production`
   - `.env.testing`

5. **Use secret management** in production:
   - AWS Secrets Manager
   - HashiCorp Vault
   - Kubernetes Secrets

### CORS Configuration

Restrict CORS origins in production:

```bash
# Development - allow local
CORS_ORIGINS=http://localhost:3000,http://localhost:5173

# Production - specific domain only
CORS_ORIGINS=https://yourdomain.com
```

### Rate Limiting

Always enable in production:

```bash
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=30  # Adjust based on needs
```

---

## Validation

### Check Required Variables

Before starting the application, verify all required variables are set:

```bash
# Backend check
python -c "from app.config import settings; print('✓ Config valid')"

# Frontend check (during build)
npm run build
```

### Variable Precedence

Variables are loaded in this order (later overrides earlier):

1. Default values in code
2. `.env` file
3. Environment variables
4. Command-line arguments (if supported)

---

## Troubleshooting

### Variable Not Loading

1. **Check file name**: Must be `.env` (or `.env.local`, `.env.production`, etc.)
2. **Check file location**: Must be in project root or appropriate directory
3. **No spaces around `=`**: Use `KEY=value`, not `KEY = value`
4. **No quotes needed** (usually): `KEY=value`, not `KEY="value"`
5. **Restart application** after changes

### Frontend Variables

Frontend variables must be prefixed with `VITE_`:
- ✅ `VITE_API_URL=http://localhost:8000`
- ❌ `API_URL=http://localhost:8000` (won't work)

Rebuild after changing frontend variables:
```bash
npm run build
```

### Docker Variables

Docker Compose reads from `.env` in the same directory as `docker-compose.yml`.

Rebuild containers after changing Docker variables:
```bash
docker-compose up --build
```

---

## Additional Resources

- [Setup Guide](SETUP.md)
- [Developer Guide](DEVELOPER_GUIDE.md)
- [Troubleshooting](TROUBLESHOOTING.md)
