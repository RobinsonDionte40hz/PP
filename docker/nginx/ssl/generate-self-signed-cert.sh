#!/bin/bash

# Generate self-signed SSL certificate for development
# Usage: ./generate-self-signed-cert.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSL_DIR="$SCRIPT_DIR"

echo "Generating self-signed SSL certificate..."
echo "Location: $SSL_DIR"

# Generate private key and certificate
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout "$SSL_DIR/key.pem" \
  -out "$SSL_DIR/cert.pem" \
  -subj "/C=US/ST=State/L=City/O=PP Development/CN=localhost" \
  -addext "subjectAltName=DNS:localhost,DNS:*.localhost,IP:127.0.0.1"

# Set proper permissions
chmod 600 "$SSL_DIR/key.pem"
chmod 644 "$SSL_DIR/cert.pem"

echo "✅ Certificate generated successfully!"
echo ""
echo "Files created:"
echo "  - Private key: $SSL_DIR/key.pem"
echo "  - Certificate: $SSL_DIR/cert.pem"
echo ""
echo "Certificate details:"
openssl x509 -in "$SSL_DIR/cert.pem" -noout -subject -dates
echo ""
echo "⚠️  This is a self-signed certificate for development only!"
echo "    Your browser will show a security warning - this is expected."
