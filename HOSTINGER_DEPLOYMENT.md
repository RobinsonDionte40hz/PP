# Hostinger VPS Deployment Guide for Protein Platform

> **🌐 Production Site**: This platform is live at **https://emergentfolds.com**

## Prerequisites

1. **Hostinger VPS** (minimum recommended: 2 vCPU, 4GB RAM, 40GB SSD)
2. **Domain** pointed to your VPS IP address
3. **SSH access** to your VPS

## Quick Deployment

### Step 1: Connect to Your VPS

From Windows PowerShell or Command Prompt:
```cmd
ssh root@YOUR_VPS_IP
```

### Step 2: Install Docker (if not installed)

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
apt install docker-compose-plugin -y
```

### Step 3: Transfer Project Files

**Option A: Using Git (recommended)**
```bash
cd /opt
git clone https://github.com/RobinsonDionte40hz/PP.git
cd PP
```

**Option B: Using SCP (from your Windows machine)**
```cmd
# In Git Bash or WSL
scp -r /c/Users/diont/OneDrive/Desktop/Projects/PP root@YOUR_VPS_IP:/opt/
```

**Option C: Using FileZilla**
1. Connect via SFTP to your VPS
2. Upload the entire PP folder to `/opt/`

### Step 4: Configure and Deploy

```bash
cd /opt/PP

# Set your domain and email (emergentfolds.com is the production domain)
export DOMAIN="emergentfolds.com"
export EMAIL="your-email@example.com"

# Make script executable and run
chmod +x deploy-hostinger.sh
./deploy-hostinger.sh
```

## Manual Deployment Steps

If you prefer manual deployment:

### 1. Create Production Environment File

```bash
cd /opt/PP

# Generate secure secrets
SECRET_KEY=$(openssl rand -hex 32)
JWT_SECRET=$(openssl rand -hex 32)
DB_PASSWORD=$(openssl rand -hex 16)

cat > .env.production << EOF
POSTGRES_USER=pp_user
POSTGRES_PASSWORD=$DB_PASSWORD
POSTGRES_DB=pp_database
SECRET_KEY=$SECRET_KEY
JWT_SECRET_KEY=$JWT_SECRET
DOMAIN=yourdomain.com
CORS_ORIGINS=https://yourdomain.com
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
JWT_ALGORITHM=HS256
REDIS_URL=redis://redis:6379/0
REDIS_SESSION_URL=redis://redis:6379/1
LOG_LEVEL=INFO
ENVIRONMENT=production
EOF
```

### 2. Build Frontend

```bash
cd /opt/PP/frontend

# Create frontend production env
cat > .env.production << EOF
VITE_API_URL=https://yourdomain.com/api
VITE_WS_URL=wss://yourdomain.com
EOF

npm install
npm run build
```

### 3. Get SSL Certificate

```bash
apt install certbot -y

# Make sure no service is using port 80
certbot certonly --standalone -d yourdomain.com -d www.yourdomain.com

# Copy certificates
mkdir -p /opt/PP/docker/nginx/ssl
cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem /opt/PP/docker/nginx/ssl/
cp /etc/letsencrypt/live/yourdomain.com/privkey.pem /opt/PP/docker/nginx/ssl/
```

### 4. Configure Nginx

```bash
# Update domain in nginx config
sed 's/YOUR_DOMAIN/yourdomain.com/g' \
    /opt/PP/docker/nginx/conf.d/production.conf.template \
    > /opt/PP/docker/nginx/conf.d/default.conf
```

### 5. Start Services

```bash
cd /opt/PP
docker compose -f docker-compose.prod.yml up -d --build
```

### 6. Verify Deployment

```bash
# Check containers are running
docker ps

# Check logs
docker logs pp_backend
docker logs pp_nginx

# Test health endpoint
curl http://localhost:8000/health
```

## DNS Configuration

In your Hostinger DNS settings, add:

| Type | Name | Value | TTL |
|------|------|-------|-----|
| A | @ | YOUR_VPS_IP | 3600 |
| A | www | YOUR_VPS_IP | 3600 |

## Firewall Configuration

```bash
# Allow HTTP, HTTPS, and SSH
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable
```

## Maintenance Commands

```bash
# View all logs
docker compose -f docker-compose.prod.yml logs -f

# View specific service logs
docker logs pp_backend -f
docker logs pp_worker -f
docker logs pp_nginx -f

# Restart all services
docker compose -f docker-compose.prod.yml restart

# Restart specific service
docker restart pp_backend

# Stop all services
docker compose -f docker-compose.prod.yml down

# Update and redeploy
git pull
docker compose -f docker-compose.prod.yml up -d --build

# Database backup
docker exec pp_postgres pg_dump -U pp_user pp_database > backup_$(date +%Y%m%d).sql
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker compose -f docker-compose.prod.yml logs backend

# Check resource usage
docker stats
```

### SSL certificate issues
```bash
# Renew certificate
certbot renew

# Copy new certificates
cp /etc/letsencrypt/live/yourdomain.com/*.pem /opt/PP/docker/nginx/ssl/
docker restart pp_nginx
```

### WebSocket connection issues
1. Ensure nginx config has WebSocket upgrade headers
2. Check CORS_ORIGINS in .env.production matches your domain

### Database connection issues
```bash
# Check postgres is running
docker logs pp_postgres

# Test connection
docker exec -it pp_postgres psql -U pp_user -d pp_database
```

## Resource Requirements

| Protein Size | Recommended VPS |
|-------------|-----------------|
| Small (<50 residues) | 1 vCPU, 2GB RAM |
| Medium (50-150) | 2 vCPU, 4GB RAM |
| Large (150+) | 4 vCPU, 8GB RAM |

## Security Checklist

- [ ] Change default SSH port (optional)
- [ ] Set up SSH key authentication
- [ ] Enable firewall (ufw)
- [ ] Keep system updated (`apt update && apt upgrade`)
- [ ] Set up automated backups
- [ ] Monitor disk space
- [ ] Set up fail2ban for brute force protection

## Support

If you encounter issues:
1. Check container logs: `docker compose logs`
2. Verify DNS propagation: https://dnschecker.org
3. Test SSL: https://www.ssllabs.com/ssltest/
