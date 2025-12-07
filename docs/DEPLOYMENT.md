# Production Deployment Guide

## Table of Contents
1. [Live Deployment Info](#live-deployment-info)
2. [Overview](#overview)
3. [Prerequisites](#prerequisites)
4. [Environment Configuration](#environment-configuration)
5. [Database Setup](#database-setup)
6. [Redis Configuration](#redis-configuration)
7. [SSL/TLS Setup](#ssltls-setup)
8. [Docker Deployment](#docker-deployment)
9. [Manual Deployment](#manual-deployment)
10. [Security Hardening](#security-hardening)
11. [Monitoring & Maintenance](#monitoring--maintenance)
12. [Backup & Recovery](#backup--recovery)
13. [Troubleshooting](#troubleshooting)

---

## Live Deployment Info

The production site **[emergentfolds.com](https://emergentfolds.com)** is deployed on:

| Setting | Value |
|---------|-------|
| **Provider** | Hostinger VPS |
| **Domain** | emergentfolds.com |
| **Server Path** | `/opt/PP` |
| **Compose File** | `docker-compose.prod.yml` |
| **SSL** | Let's Encrypt via Nginx |

### Container Names (Production)

| Service | Container Name | Port |
|---------|---------------|------|
| Nginx | `pp_nginx` | 80, 443 |
| Backend | `pp_backend` | 8000 (internal) |
| Worker | `pp_worker` | - |
| PostgreSQL | `pp_postgres` | 5432 (internal) |
| Redis | `pp_redis` | 6379 (internal) |

### Quick Operations

```bash
# SSH to VPS
ssh root@<vps-ip>
cd /opt/PP

# Check status
docker ps -a

# Restart all services
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml up -d

# View logs
docker logs pp_backend --tail 100
docker logs pp_nginx --tail 100
docker logs pp_worker --tail 100

# Check Redis health
docker exec pp_redis redis-cli ping

# Check PostgreSQL
docker exec pp_postgres pg_isready
```

### Common Issues & Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| **502 Bad Gateway** | Nginx can't reach backend | `docker compose -f docker-compose.prod.yml down && docker compose -f docker-compose.prod.yml up -d` |
| **Login fails** | Redis session issue | Check `docker logs pp_redis --tail 50` |
| **"Host unreachable"** | Container network mismatch | Full restart with prod compose file |
| **Slow dashboard** | Database or Redis overload | Check `docker logs pp_backend --tail 100` |
| **No nginx container** | Used wrong compose file | Use `docker-compose.prod.yml` not `docker-compose.yml` |

---

## Overview

This guide covers production deployment of the Protein Prediction Platform with authentication, including security best practices, configuration, and maintenance.

### System Architecture

```
┌─────────────┐
│   Nginx     │ ← SSL/TLS, Reverse Proxy
│   (Port 80) │
└──────┬──────┘
       │
       ├─────→ Frontend (Static Files)
       │
       └─────→ Backend (FastAPI)
                  │
                  ├─────→ PostgreSQL (Database)
                  │
                  ├─────→ Redis DB 0 (Celery Queue)
                  │
                  └─────→ Redis DB 1 (Auth Sessions)
```

---

## Prerequisites

### Required Software
- **Docker** 24.0+ and **Docker Compose** 2.20+ (recommended)
  - OR **Python** 3.8-3.12 (for manual deployment)
- **PostgreSQL** 13+ (production database)
- **Redis** 7+ (sessions and task queue)
- **Node.js** 18+ (for frontend build)
- **Nginx** 1.25+ (reverse proxy)

### Server Requirements
- **Minimum**: 2 CPU cores, 4 GB RAM, 20 GB storage
- **Recommended**: 4 CPU cores, 8 GB RAM, 50 GB SSD
- **OS**: Ubuntu 22.04 LTS, Debian 12, or RHEL 9

### Domain & DNS
- Domain name with DNS A record pointing to your server IP
- SSL certificate (Let's Encrypt recommended)

---

## Environment Configuration

### 1. Generate Secret Key

**Critical**: Generate a secure random secret key for JWT signing:

```bash
# Python method (recommended)
python -c "import secrets; print(secrets.token_urlsafe(32))"

# OpenSSL method
openssl rand -base64 32
```

**Save this key securely** - losing it will invalidate all sessions.

### 2. Create Production Environment File

Create `.env.production` in the project root:

```bash
# Copy template
cp .env.example .env.production

# Edit with your values
nano .env.production
```

### 3. Configure Production Variables

**CRITICAL CHANGES from development**:

```bash
# Application
APP_ENV=production
DEBUG=false
SHOW_ERROR_DETAILS=false

# Security (CHANGE THESE!)
SECRET_KEY=<your-generated-secret-key>
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15  # Shorter in production
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
JWT_ALGORITHM=HS256
SESSION_TTL_SECONDS=900  # 15 minutes

# Security Headers
ENABLE_CSRF=true
ENABLE_HSTS=true  # Only with HTTPS!
HSTS_MAX_AGE=31536000

# Database (PostgreSQL)
DATABASE_URL=postgresql://pp_user:STRONG_PASSWORD@localhost:5432/pp_db

# Redis (with password)
REDIS_URL=redis://:REDIS_PASSWORD@localhost:6379/0
REDIS_SESSION_URL=redis://:REDIS_PASSWORD@localhost:6379/1

# CORS (restrict to your domain)
CORS_ORIGINS=["https://yourdomain.com","https://www.yourdomain.com"]
CORS_ALLOW_CREDENTIALS=true

# Logging
LOG_LEVEL=WARNING
LOG_FORMAT=json
LOG_FILE=./logs/production.log

# Rate Limiting (stricter)
RATE_LIMIT_REGISTER=3
RATE_LIMIT_LOGIN=5
RATE_LIMIT_REFRESH=10
RATE_LIMIT_API=30

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=4  # CPU count
WORKER_CLASS=uvicorn.workers.UvicornWorker

# PostgreSQL (Docker)
POSTGRES_USER=pp_user
POSTGRES_PASSWORD=STRONG_PASSWORD_HERE
POSTGRES_DB=pp_db
```

### 4. Frontend Environment

Create `frontend/.env.production`:

```bash
VITE_ENV=production
VITE_API_BASE_URL=https://api.yourdomain.com
VITE_WS_BASE_URL=wss://api.yourdomain.com
VITE_DEBUG=false
VITE_ENABLE_AUTH=true
VITE_TOKEN_REFRESH_INTERVAL=840000  # 14 minutes
VITE_SESSION_TIMEOUT_WARNING=300000  # 5 minutes
```

---

## Database Setup

### PostgreSQL Production Configuration

#### 1. Install PostgreSQL

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql-15 postgresql-contrib

# Start service
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

#### 2. Create Database and User

```bash
sudo -u postgres psql

-- Create user
CREATE USER pp_user WITH PASSWORD 'your-strong-password';

-- Create database
CREATE DATABASE pp_db OWNER pp_user;

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE pp_db TO pp_user;

-- Exit
\q
```

#### 3. Configure PostgreSQL Security

Edit `/etc/postgresql/15/main/pg_hba.conf`:

```
# Allow local connections with password
local   pp_db   pp_user   scram-sha-256
host    pp_db   pp_user   127.0.0.1/32   scram-sha-256
```

Restart PostgreSQL:

```bash
sudo systemctl restart postgresql
```

#### 4. Run Migrations

```bash
cd backend
python -m alembic upgrade head
```

#### 5. Create Admin User

```bash
python -c "
from app.models.user import User
from app.database import SessionLocal
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
db = SessionLocal()

admin = User(
    email='admin@yourdomain.com',
    username='admin',
    hashed_password=pwd_context.hash('your-secure-password'),
    is_active=True
)

db.add(admin)
db.commit()
print('Admin user created!')
"
```

---

## Redis Configuration

### Production Redis Setup

#### 1. Install Redis

```bash
# Ubuntu/Debian
sudo apt install redis-server

# Configure systemd
sudo systemctl enable redis-server
```

#### 2. Secure Redis Configuration

Edit `/etc/redis/redis.conf`:

```bash
# Bind to localhost only
bind 127.0.0.1

# Enable password
requirepass YOUR_STRONG_REDIS_PASSWORD

# Disable dangerous commands
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command CONFIG ""

# Set max memory and eviction policy
maxmemory 1gb
maxmemory-policy allkeys-lru

# Enable persistence
appendonly yes
appendfsync everysec

# Disable remote connections
protected-mode yes
```

#### 3. Restart Redis

```bash
sudo systemctl restart redis-server
```

#### 4. Test Connection

```bash
redis-cli -a YOUR_STRONG_REDIS_PASSWORD ping
# Should return: PONG
```

### Redis Database Separation

- **DB 0**: Celery task queue and results
- **DB 1**: Authentication sessions (isolated from tasks)

This separation prevents session data from being affected by Celery task cleanup.

---

## SSL/TLS Setup

### Using Let's Encrypt (Recommended)

#### 1. Install Certbot

```bash
# Ubuntu/Debian
sudo apt install certbot python3-certbot-nginx
```

#### 2. Obtain Certificate

```bash
# Stop services temporarily
sudo systemctl stop nginx

# Get certificate
sudo certbot certonly --standalone -d yourdomain.com -d www.yourdomain.com

# Certificates saved to:
# /etc/letsencrypt/live/yourdomain.com/fullchain.pem
# /etc/letsencrypt/live/yourdomain.com/privkey.pem
```

#### 3. Auto-Renewal

```bash
# Test renewal
sudo certbot renew --dry-run

# Enable auto-renewal (already configured by certbot)
sudo systemctl status certbot.timer
```

### Manual SSL Certificate

If using a commercial certificate:

1. Place certificate files in `docker/nginx/ssl/`:
   - `server.crt` (certificate + intermediate chain)
   - `server.key` (private key)

2. Set proper permissions:
   ```bash
   chmod 600 docker/nginx/ssl/server.key
   chmod 644 docker/nginx/ssl/server.crt
   ```

---

## Docker Deployment

### 1. Build Images

```bash
# Build all services
docker compose -f docker-compose.prod.yml build

# Or build individually
docker compose -f docker-compose.prod.yml build backend
docker compose -f docker-compose.prod.yml build worker
```

### 2. Start Services

```bash
# Start all services
docker compose -f docker-compose.prod.yml up -d

# Check status
docker compose -f docker-compose.prod.yml ps

# View logs
docker compose -f docker-compose.prod.yml logs -f backend
```

### 3. Initialize Database

```bash
# Run migrations
docker compose -f docker-compose.prod.yml exec backend alembic upgrade head

# Create admin user (interactive)
docker compose -f docker-compose.prod.yml exec backend python scripts/create_admin.py
```

### 4. Verify Deployment

```bash
# Check backend health
curl https://yourdomain.com/api/health

# Expected response:
# {"status":"healthy","timestamp":"..."}

# Check frontend
curl -I https://yourdomain.com
# Should return 200 OK
```

### 5. Enable Monitoring (Optional)

```bash
# Start with monitoring services
docker compose -f docker-compose.prod.yml --profile monitoring up -d

# Access Grafana at http://your-server:3001
```

---

## Manual Deployment

### Backend Setup

#### 1. Install Dependencies

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. Run Migrations

```bash
alembic upgrade head
```

#### 3. Start with Gunicorn

```bash
gunicorn -k uvicorn.workers.UvicornWorker \
  -w 4 \
  -b 0.0.0.0:8000 \
  --access-logfile /var/log/pp/access.log \
  --error-logfile /var/log/pp/error.log \
  --log-level warning \
  wsgi:app
```

#### 4. Create Systemd Service

Create `/etc/systemd/system/pp-backend.service`:

```ini
[Unit]
Description=PP Backend API
After=network.target postgresql.service redis.service

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/var/www/pp/backend
Environment="PATH=/var/www/pp/backend/venv/bin"
EnvironmentFile=/var/www/pp/.env.production
ExecStart=/var/www/pp/backend/venv/bin/gunicorn \
  -k uvicorn.workers.UvicornWorker \
  -w 4 \
  -b 127.0.0.1:8000 \
  wsgi:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable pp-backend
sudo systemctl start pp-backend
```

### Frontend Setup

#### 1. Build Frontend

```bash
cd frontend
npm install
npm run build
```

#### 2. Deploy Static Files

```bash
sudo cp -r dist/* /var/www/pp/frontend/
sudo chown -R www-data:www-data /var/www/pp/frontend
```

### Celery Worker

#### 1. Create Systemd Service

Create `/etc/systemd/system/pp-celery.service`:

```ini
[Unit]
Description=PP Celery Worker
After=network.target redis.service

[Service]
Type=forking
User=www-data
Group=www-data
WorkingDirectory=/var/www/pp/backend
Environment="PATH=/var/www/pp/backend/venv/bin"
EnvironmentFile=/var/www/pp/.env.production
ExecStart=/var/www/pp/backend/venv/bin/celery -A celery_app worker \
  --loglevel=warning \
  --logfile=/var/log/pp/celery.log \
  --concurrency=4
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable pp-celery
sudo systemctl start pp-celery
```

### Nginx Configuration

Create `/etc/nginx/sites-available/pp`:

```nginx
# Rate limiting zones
limit_req_zone $binary_remote_addr zone=login:10m rate=5r/m;
limit_req_zone $binary_remote_addr zone=register:10m rate=3r/h;
limit_req_zone $binary_remote_addr zone=api:10m rate=30r/m;

# Redirect HTTP to HTTPS
server {
    listen 80;
    listen [::]:80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Frontend static files
    root /var/www/pp/frontend;
    index index.html;

    # Serve frontend
    location / {
        try_files $uri $uri/ /index.html;
        expires 1h;
        add_header Cache-Control "public, immutable";
    }

    # Backend API
    location /api {
        limit_req zone=api burst=10 nodelay;
        
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Authentication endpoints with stricter rate limits
    location /api/auth/login {
        limit_req zone=login burst=3 nodelay;
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/auth/register {
        limit_req zone=register burst=2 nodelay;
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support
    location /socket.io {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket timeouts
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

    # Health check (no rate limit)
    location /api/health {
        proxy_pass http://127.0.0.1:8000;
        access_log off;
    }

    # Static assets caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

Enable and test:

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/pp /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload
sudo systemctl reload nginx
```

---

## Security Hardening

### 1. Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp  # SSH
sudo ufw allow 80/tcp  # HTTP
sudo ufw allow 443/tcp # HTTPS
sudo ufw enable

# Deny direct access to backend
sudo ufw deny 8000/tcp
```

### 2. SSH Hardening

Edit `/etc/ssh/sshd_config`:

```
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
```

Restart SSH:

```bash
sudo systemctl restart sshd
```

### 3. Database Security

- Use strong passwords (20+ characters, mixed case, numbers, symbols)
- Restrict PostgreSQL to localhost connections
- Enable SSL for PostgreSQL connections (if remote access needed)
- Regular backups with encryption

### 4. Redis Security

- Always use password authentication
- Bind to localhost only
- Disable dangerous commands
- Use separate databases for different purposes

### 5. Application Security Checklist

- [x] Secret key is strong and random
- [x] DEBUG=false in production
- [x] HSTS enabled with HTTPS
- [x] CORS restricted to your domain
- [x] Rate limiting configured
- [x] CSRF protection enabled
- [x] JWT tokens have short expiration
- [x] Session TTL configured properly
- [x] Error details hidden from users
- [x] All passwords use bcrypt with cost 12
- [x] Redis requires authentication
- [x] PostgreSQL uses strong credentials

### 6. Regular Security Updates

```bash
# Set up automatic security updates
sudo apt install unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

---

## Monitoring & Maintenance

### Application Monitoring

#### 1. Health Checks

```bash
# Backend health
curl https://yourdomain.com/api/health

# Database connection
docker compose -f docker-compose.prod.yml exec backend python -c "
from app.database import engine
with engine.connect() as conn:
    print('Database: OK')
"

# Redis connection
redis-cli -a YOUR_PASSWORD ping
```

#### 2. Log Monitoring

```bash
# Backend logs
sudo journalctl -u pp-backend -f

# Docker logs
docker compose -f docker-compose.prod.yml logs -f backend

# Nginx access logs
sudo tail -f /var/log/nginx/access.log

# Nginx error logs
sudo tail -f /var/log/nginx/error.log
```

#### 3. Resource Monitoring

```bash
# System resources
htop

# Docker stats
docker stats

# Database size
sudo -u postgres psql -d pp_db -c "
SELECT pg_size_pretty(pg_database_size('pp_db'));
"

# Redis memory
redis-cli -a YOUR_PASSWORD INFO memory
```

### Performance Monitoring

Set up monitoring with Prometheus and Grafana (included in `docker-compose.prod.yml`):

```bash
# Start with monitoring
docker compose -f docker-compose.prod.yml --profile monitoring up -d

# Access Grafana
# URL: http://your-server:3001
# Default: admin/admin (change immediately!)
```

### Alert Configuration

Create alerts for:
- High CPU/memory usage (>80%)
- Disk space low (<10%)
- Failed login attempts (>10/minute)
- Backend health check failures
- Database connection errors
- Redis connection errors

---

## Backup & Recovery

### Database Backups

#### 1. Automated Backup Script

Create `/usr/local/bin/backup-pp-db.sh`:

```bash
#!/bin/bash
BACKUP_DIR="/backups/postgresql"
DATE=$(date +%Y%m%d_%H%M%S)
FILENAME="pp_db_${DATE}.sql.gz"

# Create backup
sudo -u postgres pg_dump pp_db | gzip > "${BACKUP_DIR}/${FILENAME}"

# Keep only last 30 days
find ${BACKUP_DIR} -name "pp_db_*.sql.gz" -mtime +30 -delete

# Log
echo "$(date): Backup created: ${FILENAME}" >> /var/log/pp/backups.log
```

Make executable:

```bash
sudo chmod +x /usr/local/bin/backup-pp-db.sh
```

#### 2. Schedule Daily Backups

Add to crontab:

```bash
sudo crontab -e

# Add this line (runs at 2 AM daily)
0 2 * * * /usr/local/bin/backup-pp-db.sh
```

#### 3. Restore from Backup

```bash
# Extract backup
gunzip pp_db_20250120_020000.sql.gz

# Restore
sudo -u postgres psql pp_db < pp_db_20250120_020000.sql
```

### Redis Backups

Redis automatically saves to disk with AOF persistence. To manually backup:

```bash
# Trigger save
redis-cli -a YOUR_PASSWORD BGSAVE

# Copy files
sudo cp /var/lib/redis/dump.rdb /backups/redis/dump_$(date +%Y%m%d).rdb
sudo cp /var/lib/redis/appendonly.aof /backups/redis/appendonly_$(date +%Y%m%d).aof
```

### Application Data Backups

```bash
# Backup important directories
tar -czf /backups/pp_data_$(date +%Y%m%d).tar.gz \
  /var/www/pp/checkpoints \
  /var/www/pp/results \
  /var/www/pp/pdb_cache
```

### Disaster Recovery Plan

1. **Database Failure**: Restore from most recent PostgreSQL backup
2. **Redis Failure**: Redis will rebuild from AOF file on restart
3. **Application Failure**: Restart services, check logs, restore from backup if needed
4. **Server Failure**: Provision new server, restore from backups, update DNS

**Recovery Time Objective (RTO)**: 4 hours  
**Recovery Point Objective (RPO)**: 24 hours (daily backups)

---

## Troubleshooting

### Common Issues

#### 1. Backend Won't Start

```bash
# Check logs
docker compose -f docker-compose.prod.yml logs backend

# Common causes:
# - Secret key not set
# - Database connection failed
# - Redis connection failed
# - Port 8000 already in use
```

**Solutions**:
- Verify `.env.production` is configured
- Test database: `psql postgresql://user:pass@localhost/pp_db`
- Test Redis: `redis-cli -a PASSWORD ping`
- Check port: `sudo lsof -i :8000`

#### 2. Authentication Failures

```bash
# Check Redis session database
redis-cli -a PASSWORD
SELECT 1
KEYS "session:*"
```

**Solutions**:
- Verify `REDIS_SESSION_URL` is set to DB 1
- Check JWT secret key is correct
- Verify token expiration settings
- Clear Redis sessions: `redis-cli -a PASSWORD -n 1 FLUSHDB`

#### 3. CORS Errors

```bash
# Check backend logs for CORS errors
docker compose logs backend | grep CORS
```

**Solutions**:
- Verify `CORS_ORIGINS` includes your domain
- Check protocol (http vs https) matches
- Ensure `CORS_ALLOW_CREDENTIALS=true`

#### 4. SSL Certificate Errors

```bash
# Check certificate
sudo certbot certificates

# Test SSL configuration
sudo nginx -t
```

**Solutions**:
- Renew certificate: `sudo certbot renew`
- Check certificate paths in nginx config
- Verify DNS is pointing to correct server

#### 5. High Memory Usage

```bash
# Check memory usage
free -h
docker stats

# Check Redis memory
redis-cli -a PASSWORD INFO memory
```

**Solutions**:
- Increase server RAM
- Configure Redis maxmemory
- Reduce Celery worker concurrency
- Check for memory leaks in logs

#### 6. Slow Performance

```bash
# Check database queries
docker compose exec backend python -c "
from app.database import engine
engine.echo = True
# Run slow query
"

# Check Redis latency
redis-cli -a PASSWORD --latency
```

**Solutions**:
- Add database indexes
- Enable query caching
- Increase worker count
- Use connection pooling
- Monitor slow queries

### Getting Help

1. **Check Logs**: Always start with logs
   ```bash
   # All logs
   docker compose -f docker-compose.prod.yml logs

   # Specific service
   docker compose logs backend
   ```

2. **Check Documentation**:
   - `/docs/SETUP.md` - Installation guide
   - `/docs/API.md` - API reference
   - `/docs/AUTHENTICATION_FLOWS.md` - Auth diagrams
   - `/docs/TROUBLESHOOTING.md` - Detailed troubleshooting

3. **Enable Debug Logging** (temporarily):
   ```bash
   # Edit .env.production
   LOG_LEVEL=DEBUG

   # Restart
   docker compose -f docker-compose.prod.yml restart backend
   ```

4. **Health Check Script**:
   ```bash
   #!/bin/bash
   echo "=== Health Check ==="
   
   echo "Backend:"
   curl -f https://yourdomain.com/api/health || echo "FAILED"
   
   echo "Database:"
   sudo -u postgres psql -d pp_db -c "SELECT 1;" || echo "FAILED"
   
   echo "Redis:"
   redis-cli -a PASSWORD ping || echo "FAILED"
   
   echo "Disk Space:"
   df -h | grep -E "/$|/var"
   
   echo "Memory:"
   free -h
   ```

---

## Production Checklist

Before going live, verify:

### Security
- [ ] Secret key changed from default
- [ ] All passwords are strong (20+ characters)
- [ ] DEBUG=false
- [ ] SHOW_ERROR_DETAILS=false
- [ ] HSTS enabled
- [ ] SSL certificate installed and valid
- [ ] Firewall configured
- [ ] SSH hardened (no root login, key-only)
- [ ] CORS restricted to your domain
- [ ] Rate limiting enabled
- [ ] Redis password protected
- [ ] PostgreSQL uses strong credentials

### Configuration
- [ ] APP_ENV=production
- [ ] JWT expiration times configured
- [ ] Session TTL configured
- [ ] Database connection pooling enabled
- [ ] Redis DB separation configured
- [ ] Worker count matches CPU cores
- [ ] Log level set to WARNING or ERROR
- [ ] Backup schedule configured

### Infrastructure
- [ ] SSL certificate auto-renewal working
- [ ] Database backups automated
- [ ] Monitoring configured
- [ ] Alerts configured
- [ ] Log rotation configured
- [ ] Disk space monitoring active
- [ ] Systemd services enabled
- [ ] Health checks passing

### Testing
- [ ] Backend health endpoint responding
- [ ] Authentication flow working
- [ ] Registration working
- [ ] Login working
- [ ] Token refresh working
- [ ] Protected endpoints secured
- [ ] CORS working correctly
- [ ] WebSocket connections working
- [ ] 3D visualization loading
- [ ] Predictions completing successfully

### Documentation
- [ ] Admin credentials documented securely
- [ ] Backup procedures documented
- [ ] Recovery procedures documented
- [ ] Monitoring dashboard configured
- [ ] On-call procedures defined

---

## Additional Resources

- **FastAPI Security**: https://fastapi.tiangolo.com/tutorial/security/
- **PostgreSQL Hardening**: https://www.postgresql.org/docs/current/auth-pg-hba-conf.html
- **Redis Security**: https://redis.io/docs/management/security/
- **Nginx Hardening**: https://www.nginx.com/blog/hardening-nginx-http-server/
- **Let's Encrypt**: https://letsencrypt.org/getting-started/
- **Docker Production**: https://docs.docker.com/engine/security/

---

**Document Version**: 1.0.0  
**Last Updated**: January 2025  
**Maintainer**: PP Development Team
