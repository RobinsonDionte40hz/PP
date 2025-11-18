# API Security Implementation

## ✅ What's Protected

### 🚦 Rate Limiting (Prevents Abuse)
All API endpoints are now rate-limited to prevent abuse and server overload:

| Endpoint | Limit | Reason |
|----------|-------|--------|
| `POST /api/predictions` | **10/minute** | Compute-intensive (runs predictions) |
| `GET /api/predictions` | **30/minute** | Database queries |
| `GET /api/predictions/{id}` | **60/minute** | Simple reads |
| Other endpoints | **100/minute** | Default protection |

**How it works:**
- Tracks requests by IP address
- Returns HTTP 429 (Too Many Requests) when exceeded
- Resets every minute

**Adjusting limits:** Edit `backend/app/security.py` → `SecurityConfig` class

---

### 🛡️ Input Validation (Prevents Crashes & Injection)

#### **Sequence Validation**
✅ **Length limits:**
- Minimum: 3 amino acids
- Maximum: 1000 amino acids (prevents server overload)

✅ **Character validation:**
- Only standard 20 amino acids: `ACDEFGHIKLMNPQRSTVWY`
- Auto-converts to uppercase
- Strips whitespace

✅ **Security checks:**
- Blocks SQL injection patterns (`UNION`, `SELECT`, `--`, etc.)
- Blocks script injection (`<script>`, `javascript:`, etc.)
- Blocks excessive repetition (DoS prevention)
- Sanitizes special characters

#### **Configuration Validation**
✅ **Safe ranges:**
- Iterations: 100 - 10,000
- Agents: 1 - 100
- Checkpoint interval: 10 - 1,000

✅ **Enum validation:**
- Diversity: `cautious`, `balanced`, `aggressive`
- QCPP config: `default`, `high_performance`, `high_accuracy`

---

## 📋 Configuration

All security settings are centralized in `backend/app/security.py`:

```python
class SecurityConfig:
    # Rate Limits
    RATE_LIMIT_CREATE_PREDICTION = "10/minute"
    RATE_LIMIT_LIST_PREDICTIONS = "30/minute"
    
    # Sequence Limits
    MAX_SEQUENCE_LENGTH = 1000
    MIN_SEQUENCE_LENGTH = 3
    
    # Configuration Limits
    MAX_ITERATIONS = 10000
    MAX_AGENTS = 100
    # ... and more
```

**To adjust limits:** Edit values in `SecurityConfig` class and restart server.

---

## 🔒 What's NOT Protected (Intentionally Open)

### ✅ Algorithms & Math
**All scientific code remains open and transparent:**
- QCPP quantum coherence formulas
- UBF consciousness parameters
- Golden ratio calculations
- Energy functions
- Physics integrations

**Why?** Scientific credibility requires reproducibility. Open algorithms = peer reviewable = trustworthy.

---

## 🚀 Production Deployment Recommendations

### **Essential (Already Implemented):**
✅ Rate limiting
✅ Input validation
✅ Error handling
✅ CORS configuration

### **Next Steps for Production:**

#### 1. **HTTPS Only** (Automatic on most platforms)
- Vercel/Railway/Render provide free SSL
- Never use HTTP in production

#### 2. **Environment Variables** (Already configured)
```env
# backend/.env
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
SECRET_KEY=your-secret-key-here
APP_ENV=production
```

#### 3. **API Authentication** (Optional - if charging users)
```python
# Add to main.py for API key auth
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in valid_keys:
        raise HTTPException(401, "Invalid API key")

# Apply to endpoints
@router.post("/api/predictions", dependencies=[Depends(verify_api_key)])
```

#### 4. **Database Backups**
- Automated backups (built-in on Railway/Render/AWS)
- Daily snapshots recommended

#### 5. **Monitoring** (Optional)
- Sentry for error tracking
- Datadog/New Relic for performance
- CloudWatch if using AWS

---

## 🧪 Testing Security

### Test Rate Limiting:
```bash
# Send 15 requests rapidly (should block after 10)
for i in {1..15}; do
  curl -X POST http://localhost:8000/api/predictions \
    -H "Content-Type: application/json" \
    -d '{"sequence": "ACDEFGH"}' &
done
```

### Test Invalid Sequences:
```bash
# Too short
curl -X POST http://localhost:8000/api/predictions \
  -d '{"sequence": "AC"}'
# Returns 400: "Sequence too short"

# Invalid characters
curl -X POST http://localhost:8000/api/predictions \
  -d '{"sequence": "ACDEFGH123"}'
# Returns 400: "Invalid amino acids"

# SQL injection attempt
curl -X POST http://localhost:8000/api/predictions \
  -d '{"sequence": "AC; DROP TABLE users--"}'
# Returns 400: "Sequence contains suspicious patterns"
```

---

## 📊 Monitoring Rate Limits

Check logs for rate limit violations:
```bash
# In production logs, look for:
# "Rate limit exceeded for /api/predictions"
```

Adjust limits based on actual usage patterns.

---

## 🔧 Troubleshooting

### "429 Too Many Requests"
**User hit rate limit**
- Normal behavior - limits are working
- User should wait 1 minute
- If legitimate user, increase limits in `security.py`

### "400 Bad Request: Sequence too long"
**Sequence exceeds 1000 amino acids**
- Intentional limit (prevents crashes)
- For larger proteins, increase `MAX_SEQUENCE_LENGTH` in `security.py`
- Consider splitting into domains

### Rate Limiting Not Working
**Check slowapi installation:**
```bash
pip list | grep slowapi
# Should show: slowapi 0.1.9
```

**Verify limiter in main.py:**
```python
# Should see in app/main.py:
from slowapi import Limiter
app.state.limiter = limiter
```

---

## 🎯 Summary

### ✅ **Protected:**
- API endpoints (rate limited)
- Server resources (input validation)
- Against injection attacks (sanitization)
- Against DoS (length limits, repetition detection)

### ✅ **Open & Transparent:**
- All scientific algorithms
- Mathematical formulas
- Research contributions
- Source code (for peer review)

### ⚡ **Result:**
**Secure production API that maintains scientific credibility and reproducibility.**

---

## 📚 Further Reading

- [slowapi documentation](https://github.com/laurentS/slowapi)
- [FastAPI security best practices](https://fastapi.tiangolo.com/tutorial/security/)
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
