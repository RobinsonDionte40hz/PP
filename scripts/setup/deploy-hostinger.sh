#!/bin/bash
# Hostinger VPS Deployment Script for Protein Platform
# Run this on your Hostinger VPS after transferring project files

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Protein Platform - Hostinger Deployment${NC}"
echo -e "${GREEN}========================================${NC}"

# Configuration - EDIT THESE VALUES
DOMAIN="${DOMAIN:-yourdomain.com}"
PROJECT_DIR="${PROJECT_DIR:-/opt/PP}"
EMAIL="${EMAIL:-your-email@example.com}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run as root (use sudo)${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Step 1: Installing prerequisites...${NC}"
apt update
apt install -y curl git docker.io docker-compose-plugin certbot || true

# Node.js is optional for Docker deployment - skip if conflicts
apt install -y nodejs npm 2>/dev/null || echo "Node.js already installed or conflicts - skipping (not required for Docker deployment)"

# Start Docker
systemctl start docker
systemctl enable docker

echo -e "\n${YELLOW}Step 2: Checking project files...${NC}"
if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}Project directory not found at $PROJECT_DIR${NC}"
    echo "Please transfer your project files first."
    exit 1
fi

cd "$PROJECT_DIR"

echo -e "\n${YELLOW}Step 3: Checking environment file...${NC}"
if [ ! -f ".env.production" ]; then
    echo -e "${RED}.env.production not found!${NC}"
    echo "Creating template..."
    
    # Generate random secrets
    SECRET_KEY=$(openssl rand -hex 32)
    JWT_SECRET=$(openssl rand -hex 32)
    DB_PASSWORD=$(openssl rand -hex 16)
    
    cat > .env.production << EOF
# Database Configuration
POSTGRES_USER=pp_user
POSTGRES_PASSWORD=$DB_PASSWORD
POSTGRES_DB=pp_database

# Security Keys (auto-generated)
SECRET_KEY=$SECRET_KEY
JWT_SECRET_KEY=$JWT_SECRET

# Domain Configuration
DOMAIN=$DOMAIN
CORS_ORIGINS=https://$DOMAIN

# JWT Settings
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
JWT_ALGORITHM=HS256

# Redis
REDIS_URL=redis://redis:6379/0
REDIS_SESSION_URL=redis://redis:6379/1

# Logging
LOG_LEVEL=INFO
ENVIRONMENT=production
EOF
    
    echo -e "${GREEN}Created .env.production with auto-generated secrets${NC}"
    echo -e "${YELLOW}IMPORTANT: Save these credentials securely!${NC}"
fi

echo -e "\n${YELLOW}Step 4: Building frontend...${NC}"
cd "$PROJECT_DIR/frontend"
if [ ! -d "node_modules" ]; then
    npm install
fi

# Create production env for frontend
cat > .env.production << EOF
VITE_API_URL=https://$DOMAIN/api
VITE_WS_URL=wss://$DOMAIN
EOF

npm run build

echo -e "\n${YELLOW}Step 5: Setting up SSL certificates...${NC}"
mkdir -p "$PROJECT_DIR/docker/nginx/ssl"

# Check if certificates already exist
if [ ! -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]; then
    echo "Obtaining SSL certificate from Let's Encrypt..."
    
    # Stop any services using port 80
    docker compose -f docker-compose.prod.yml down 2>/dev/null || true
    
    certbot certonly --standalone \
        --non-interactive \
        --agree-tos \
        --email "$EMAIL" \
        -d "$DOMAIN" \
        -d "www.$DOMAIN"
fi

# Copy certificates
cp "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" "$PROJECT_DIR/docker/nginx/ssl/"
cp "/etc/letsencrypt/live/$DOMAIN/privkey.pem" "$PROJECT_DIR/docker/nginx/ssl/"

echo -e "\n${YELLOW}Step 6: Configuring Nginx for your domain...${NC}"
# Update nginx config with actual domain
sed "s/YOUR_DOMAIN/$DOMAIN/g" \
    "$PROJECT_DIR/docker/nginx/conf.d/production.conf.template" \
    > "$PROJECT_DIR/docker/nginx/conf.d/default.conf"

echo -e "\n${YELLOW}Step 7: Starting services with Docker Compose...${NC}"
cd "$PROJECT_DIR"
docker compose -f docker-compose.prod.yml down 2>/dev/null || true
docker compose -f docker-compose.prod.yml up -d --build

echo -e "\n${YELLOW}Step 8: Waiting for services to start...${NC}"
sleep 30

echo -e "\n${YELLOW}Step 9: Checking service health...${NC}"
docker ps

# Check backend health
echo -e "\n${YELLOW}Checking backend health...${NC}"
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo -e "${GREEN}Backend is healthy!${NC}"
        break
    fi
    echo "Waiting for backend... ($i/10)"
    sleep 5
done

echo -e "\n${YELLOW}Step 10: Setting up SSL certificate auto-renewal...${NC}"
# Add cron job for certificate renewal
(crontab -l 2>/dev/null; echo "0 3 * * * certbot renew --quiet && cp /etc/letsencrypt/live/$DOMAIN/*.pem $PROJECT_DIR/docker/nginx/ssl/ && docker restart pp_nginx") | crontab -

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\nYour application should now be available at:"
echo -e "  ${GREEN}https://$DOMAIN${NC}"
echo -e "\nUseful commands:"
echo -e "  View logs:     docker compose -f docker-compose.prod.yml logs -f"
echo -e "  Restart:       docker compose -f docker-compose.prod.yml restart"
echo -e "  Stop:          docker compose -f docker-compose.prod.yml down"
echo -e "  Update:        git pull && docker compose -f docker-compose.prod.yml up -d --build"
echo -e "\n${YELLOW}IMPORTANT: Make sure to:${NC}"
echo -e "  1. Point your domain DNS A record to this server's IP"
echo -e "  2. Save your .env.production credentials securely"
echo -e "  3. Set up regular database backups"
