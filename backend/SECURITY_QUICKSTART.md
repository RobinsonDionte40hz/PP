# 🚀 Quick Start: Security-Enabled API

Your API now has **rate limiting** and **input validation** enabled!

## ✅ What Changed

### 1. **Rate Limiting Added**
- Prevents API abuse
- 10 predictions/minute per IP
- 30 list requests/minute per IP

### 2. **Input Validation Enhanced**
- Blocks sequences > 1000 amino acids (prevents crashes)
- Blocks SQL/script injection attempts
- Blocks excessive repetition (DoS prevention)
- Auto-sanitizes input (uppercase, trim whitespace)

### 3. **Configuration Limits**
- Iterations: 100-10,000
- Agents: 1-100
- All validated automatically

## 🏃 Running the Server

**Nothing changes! Start as usual:**

```bash
# Option 1: Using batch file
cd backend
start_backend.bat

# Option 2: Direct command
uvicorn wsgi:socket_app --reload --host 0.0.0.0 --port 8000
```

The security features are **automatic** - no configuration needed.

## 🧪 Testing Security

### Test Valid Request:
```bash
curl -X POST http://localhost:8000/api/predictions \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ACDEFGHIKLMNPQRSTVWY"}'

# ✅ Returns 201 Created
```

### Test Rate Limiting:
```bash
# Send 15 requests rapidly
for i in {1..15}; do
  curl -X POST http://localhost:8000/api/predictions \
    -H "Content-Type: application/json" \
    -d "{\"sequence\": \"ACDEFGH\"}" &
done

# ❌ Last 5 return 429 Too Many Requests
```

### Test Invalid Input:
```bash
# Too short
curl -X POST http://localhost:8000/api/predictions \
  -d '{"sequence": "AC"}'
# ❌ Returns 422: "Sequence too short"

# Invalid characters
curl -X POST http://localhost:8000/api/predictions \
  -d '{"sequence": "ACDEFGH123"}'
# ❌ Returns 422: "Invalid amino acids"

# Too many iterations
curl -X POST http://localhost:8000/api/predictions \
  -d '{"sequence": "ACDEFGH", "configuration": {"iterations": 20000}}'
# ❌ Returns 422: Validation error
```

## ⚙️ Adjusting Limits

Edit `backend/app/security.py`:

```python
class SecurityConfig:
    # Change these values as needed
    RATE_LIMIT_CREATE_PREDICTION = "10/minute"  # Increase for more traffic
    MAX_SEQUENCE_LENGTH = 1000                  # Increase for larger proteins
    MAX_ITERATIONS = 10000                      # Increase for longer runs
```

Then restart the server.

## 📊 What's Protected vs. Open

### ✅ Protected (Security):
- API endpoints (rate limited)
- Server resources (input validated)
- Against malicious input (sanitized)

### ✅ Open (Transparency):
- All algorithms (QCPP, UBF)
- Math formulas (golden ratio, etc.)
- Source code (for peer review)

## 🚢 Ready to Ship

Your API is now production-ready with:
- ✅ Rate limiting
- ✅ Input validation
- ✅ Error handling
- ✅ Security checks

**Next step:** Deploy to Vercel/Railway/Render for public access!

---

See `SECURITY.md` for full documentation.
