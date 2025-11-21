# SSL/TLS Certificate Setup

This directory contains SSL/TLS certificates for HTTPS.

## Development (Self-Signed Certificate)

Generate a self-signed certificate for development:

```bash
# Generate self-signed certificate (valid for 365 days)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout key.pem \
  -out cert.pem \
  -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"
```

## Production (Let's Encrypt)

For production, use Let's Encrypt with Certbot:

### Initial Setup

1. **Update app.conf** to temporarily serve only HTTP:
   ```bash
   # Comment out the HTTPS server block in app.conf
   ```

2. **Start Nginx**:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d nginx
   ```

3. **Run Certbot**:
   ```bash
   docker run -it --rm \
     -v $(pwd)/docker/nginx/ssl:/etc/letsencrypt \
     -v $(pwd)/docker/nginx/www:/var/www/certbot \
     certbot/certbot certonly \
     --webroot \
     --webroot-path=/var/www/certbot \
     --email your-email@example.com \
     --agree-tos \
     --no-eff-email \
     -d your-domain.com
   ```

4. **Link certificates**:
   ```bash
   # Create symlinks to the Let's Encrypt certificates
   ln -sf /etc/letsencrypt/live/your-domain.com/fullchain.pem cert.pem
   ln -sf /etc/letsencrypt/live/your-domain.com/privkey.pem key.pem
   ```

5. **Uncomment HTTPS block** in app.conf and restart:
   ```bash
   docker-compose -f docker-compose.prod.yml restart nginx
   ```

### Auto-Renewal

Add a cron job to renew certificates automatically:

```bash
# Run every day at 2 AM
0 2 * * * docker run --rm -v $(pwd)/docker/nginx/ssl:/etc/letsencrypt -v $(pwd)/docker/nginx/www:/var/www/certbot certbot/certbot renew --quiet && docker-compose -f docker-compose.prod.yml restart nginx
```

## Testing

Test your SSL configuration:
```bash
# Using curl
curl -k https://localhost

# Using OpenSSL
openssl s_client -connect localhost:443

# Check certificate expiry
openssl x509 -in cert.pem -noout -dates
```

## Security Best Practices

1. **Permissions**: Ensure private keys are not world-readable
   ```bash
   chmod 600 key.pem
   chmod 644 cert.pem
   ```

2. **Backup**: Keep encrypted backups of your certificates

3. **Monitoring**: Set up alerts for certificate expiry (30 days before)

4. **Rotation**: Rotate certificates regularly (Let's Encrypt renews every 60 days)

## Troubleshooting

### Certificate Not Found
- Check that cert.pem and key.pem exist in this directory
- Verify file permissions

### SSL Handshake Errors
- Verify certificate validity: `openssl x509 -in cert.pem -text -noout`
- Check nginx error logs: `docker logs pp_nginx`

### Let's Encrypt Rate Limits
- Production: 50 certificates per registered domain per week
- Staging: Use `--staging` flag for testing
