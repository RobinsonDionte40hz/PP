# Security Enhancement Plan - EmergentFolds

**Created**: December 7, 2025  
**Status**: In Progress  
**Priority**: High

---

## Executive Summary

This document outlines the implementation plan for enhancing security and usage control on EmergentFolds. The primary goals are:

1. **Prevent resource abuse** — Limit how many predictions users can run
2. **Reduce account farming** — Make it harder to create fake accounts
3. **Improve authentication** — Add OAuth for better security and UX

---

## Implementation Progress

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1: Usage Quotas | ✅ **COMPLETE** | Daily/monthly prediction limits |
| Phase 2: Email Verification | 🔲 Not Started | Require email verification |
| Phase 3: CAPTCHA | 🔲 Not Started | Bot protection on registration |
| Phase 4: OAuth | 🔲 Not Started | Google/GitHub social login |
| Phase 5: Frontend | 🔲 Not Started | UI for quotas and OAuth |

---

## Current Security Status: ✅ STRONG

| Security Layer | Status | Notes |
|----------------|--------|-------|
| JWT Authentication | ✅ Implemented | 30-min access, 7-day refresh tokens |
| Session Management | ✅ Implemented | Redis-backed, single-session enforcement |
| Rate Limiting | ✅ Implemented | 10/min predictions, tiered by endpoint |
| Input Validation | ✅ Implemented | SQL injection, XSS, DoS protection |
| Security Headers | ✅ Implemented | CSP, HSTS, X-Frame-Options, etc. |
| CSRF Protection | ✅ Implemented | Token-based, 1-hour rotation |
| Audit Logging | ✅ Implemented | Request logging, no sensitive data |

---

## Gap Analysis

### ❌ Missing: Usage Quotas

**Problem**: Users can run unlimited predictions over time. Rate limiting only prevents burst abuse (10/minute), not sustained abuse.

**Impact**: Server costs, resource exhaustion, unfair usage.

**Solution**: Daily/monthly prediction quotas per user.

### ❌ Missing: Email Verification

**Problem**: Users can register with fake emails and start using the platform immediately.

**Impact**: Easy account farming, spam accounts.

**Solution**: Require email verification before allowing predictions.

### ❌ Missing: OAuth/Social Login

**Problem**: Username/password only authentication.

**Impact**: Higher friction for users, no third-party identity verification.

**Solution**: Add Google and GitHub OAuth options.

### ❌ Missing: CAPTCHA

**Problem**: Registration endpoint only has rate limiting.

**Impact**: Automated bot registrations possible.

**Solution**: Add reCAPTCHA v3 or hCaptcha to registration.

---

## Implementation Plan

### Phase 1: Usage Quota System (Week 1) ✅ COMPLETE

**Goal**: Prevent unlimited resource usage

**Implemented Files**:
- `backend/app/models/user.py` - Added quota fields to User model
- `backend/app/services/quota_service.py` - Quota checking and management service
- `backend/app/api/users.py` - User API with quota endpoints
- `backend/app/tasks/quota_tasks.py` - Celery tasks for daily/monthly reset
- `backend/alembic/versions/002_add_user_quotas.py` - Database migration
- `backend/tests/test_quota_service.py` - Unit tests (18 passing)

#### Task 1.1: Add Quota Fields to User Model ✅

Added to `backend/app/models/user.py`:

```python
# Quota tracking
daily_prediction_count: int = 0
monthly_prediction_count: int = 0
daily_quota_reset_at: datetime = None
monthly_quota_reset_at: datetime = None

# Tier settings
account_tier: str = "free"  # free, pro, enterprise
daily_prediction_limit: int = 20  # Default for free tier
monthly_prediction_limit: int = 100  # Default for free tier
```

**Tier Limits**:
| Tier | Daily Limit | Monthly Limit | Price |
|------|-------------|---------------|-------|
| Free | 20 | 100 | $0 |
| Pro | 100 | 500 | TBD |
| Enterprise | Unlimited | Unlimited | Custom |

#### Task 1.2: Create Quota Checking Middleware

Create `backend/app/middleware/quota.py`:

- Check user's current count vs limit before prediction creation
- Return 429 with clear message when quota exceeded
- Include quota status in response headers

#### Task 1.3: Add Daily Reset Celery Task

Create `backend/app/tasks/quota_tasks.py`:

- Run at midnight UTC
- Reset `daily_prediction_count` for all users
- Reset `monthly_prediction_count` on 1st of month
- Log reset operations

**Database Migration**: Create Alembic migration for new fields.

---

### Phase 2: Email Verification (Week 2)

**Goal**: Prevent account farming with fake emails

#### Task 2.1: Add Email Verification Fields

Add to User model:
```python
email_verified: bool = False
email_verification_token: str = None
email_verification_sent_at: datetime = None
```

#### Task 2.2: Create Verification Endpoints

- `POST /api/auth/send-verification` — Send/resend verification email
- `GET /api/auth/verify-email/{token}` — Verify email with token
- `POST /api/auth/resend-verification` — Resend verification email

#### Task 2.3: Block Unverified Users

- Modify `require_auth_with_session` to check `email_verified`
- Allow: login, logout, resend-verification, account settings
- Block: predictions, sessions, all other features

#### Task 2.4: Email Templates

- Create HTML email template for verification
- Include: verification link, expiration time (24 hours), support contact

**SMTP Configuration**: Already exists in `config.py` (SMTP_HOST, etc.)

---

### Phase 3: CAPTCHA Integration (Week 2)

**Goal**: Prevent automated bot registrations

#### Task 3.1: Backend Integration

- Add reCAPTCHA v3 or hCaptcha verification
- Verify token on registration endpoint
- Add to `.env`: `RECAPTCHA_SITE_KEY`, `RECAPTCHA_SECRET_KEY`

#### Task 3.2: Frontend Integration

- Add CAPTCHA widget to registration form
- Pass token with registration request

**Recommendation**: Use reCAPTCHA v3 (invisible) for better UX.

---

### Phase 4: OAuth Integration (Week 3)

**Goal**: Add social login options

#### Task 4.1: Install Dependencies

```bash
pip install authlib httpx
```

#### Task 4.2: Google OAuth

Create `backend/app/services/oauth_service.py`:

- Configure Google OAuth client
- Endpoints:
  - `GET /api/auth/google` — Redirect to Google
  - `GET /api/auth/google/callback` — Handle callback
- Link Google account to existing user or create new

**Environment Variables**:
```env
GOOGLE_CLIENT_ID=your-client-id
GOOGLE_CLIENT_SECRET=your-client-secret
GOOGLE_REDIRECT_URI=https://emergentfolds.com/api/auth/google/callback
```

#### Task 4.3: GitHub OAuth

Similar to Google:
- `GET /api/auth/github` — Redirect to GitHub
- `GET /api/auth/github/callback` — Handle callback

**Environment Variables**:
```env
GITHUB_CLIENT_ID=your-client-id
GITHUB_CLIENT_SECRET=your-client-secret
GITHUB_REDIRECT_URI=https://emergentfolds.com/api/auth/github/callback
```

#### Task 4.4: Account Linking

- Allow existing users to link OAuth accounts
- Allow OAuth users to set password for traditional login
- Handle email conflicts (same email, different provider)

---

### Phase 5: Frontend Updates (Week 3-4)

**Goal**: Expose new features to users

#### Task 5.1: Quota Display

- Show quota status in dashboard header
- "5/10 predictions today"
- Warning when approaching limit
- Upgrade prompt when limit reached

#### Task 5.2: Email Verification UI

- Show banner for unverified users
- Verification success/error pages
- Resend verification button

#### Task 5.3: OAuth Buttons

- Add "Sign in with Google" button
- Add "Sign in with GitHub" button
- Account settings: link/unlink OAuth accounts

#### Task 5.4: CAPTCHA Widget

- Invisible reCAPTCHA on registration form

---

## Database Migrations

### Migration 1: Add Quota Fields

```python
# alembic/versions/xxx_add_quota_fields.py

def upgrade():
    op.add_column('users', sa.Column('daily_prediction_count', sa.Integer(), default=0))
    op.add_column('users', sa.Column('monthly_prediction_count', sa.Integer(), default=0))
    op.add_column('users', sa.Column('daily_quota_reset_at', sa.DateTime(timezone=True)))
    op.add_column('users', sa.Column('monthly_quota_reset_at', sa.DateTime(timezone=True)))
    op.add_column('users', sa.Column('account_tier', sa.String(20), default='free'))
    op.add_column('users', sa.Column('daily_prediction_limit', sa.Integer(), default=20))
    op.add_column('users', sa.Column('monthly_prediction_limit', sa.Integer(), default=100))
```

### Migration 2: Add Email Verification Fields

```python
# alembic/versions/xxx_add_email_verification.py

def upgrade():
    op.add_column('users', sa.Column('email_verified', sa.Boolean(), default=False))
    op.add_column('users', sa.Column('email_verification_token', sa.String(64)))
    op.add_column('users', sa.Column('email_verification_sent_at', sa.DateTime(timezone=True)))
```

### Migration 3: Add OAuth Fields

```python
# alembic/versions/xxx_add_oauth_fields.py

def upgrade():
    op.add_column('users', sa.Column('google_id', sa.String(100)))
    op.add_column('users', sa.Column('github_id', sa.String(100)))
    op.add_column('users', sa.Column('oauth_provider', sa.String(20)))  # 'google', 'github', None
    
    # Make password optional for OAuth users
    op.alter_column('users', 'password_hash', nullable=True)
```

---

## Environment Variables to Add

```env
# Quota Settings
DEFAULT_DAILY_QUOTA=20
DEFAULT_MONTHLY_QUOTA=100
PRO_DAILY_QUOTA=100
PRO_MONTHLY_QUOTA=500

# Email Verification
EMAIL_VERIFICATION_EXPIRE_HOURS=24
REQUIRE_EMAIL_VERIFICATION=true

# reCAPTCHA
RECAPTCHA_SITE_KEY=your-site-key
RECAPTCHA_SECRET_KEY=your-secret-key
RECAPTCHA_ENABLED=true

# Google OAuth
GOOGLE_CLIENT_ID=your-client-id
GOOGLE_CLIENT_SECRET=your-client-secret
GOOGLE_REDIRECT_URI=https://emergentfolds.com/api/auth/google/callback

# GitHub OAuth
GITHUB_CLIENT_ID=your-client-id
GITHUB_CLIENT_SECRET=your-client-secret
GITHUB_REDIRECT_URI=https://emergentfolds.com/api/auth/github/callback
```

---

## Testing Plan

### Unit Tests

- [ ] Quota increment on prediction creation
- [ ] Quota block when limit exceeded
- [ ] Daily/monthly reset logic
- [ ] Email verification token generation
- [ ] Email verification flow
- [ ] CAPTCHA validation
- [ ] OAuth token exchange
- [ ] OAuth account linking

### Integration Tests

- [ ] Full registration → verification → prediction flow
- [ ] Quota enforcement end-to-end
- [ ] OAuth login flow
- [ ] Rate limiting + quota interaction

### Manual Testing

- [ ] Test all OAuth providers with real accounts
- [ ] Test email delivery (check spam folders)
- [ ] Test CAPTCHA UX on various devices
- [ ] Test quota display in frontend

---

## Rollout Plan

### Stage 1: Soft Launch (Internal)

- Deploy to staging environment
- Test with team accounts
- Monitor for issues

### Stage 2: Gradual Rollout

- Enable email verification for new accounts only
- Existing accounts grandfathered (but encouraged to verify)
- Monitor registration/verification rates

### Stage 3: Full Enforcement

- Require email verification for all users
- Enable CAPTCHA
- Enable quotas (with generous initial limits)

### Stage 4: OAuth Launch

- Add OAuth buttons to login page
- Announce via email to existing users

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Fake account registrations | Unknown | < 5% |
| Prediction abuse incidents | Unknown | Zero |
| User registration completion | ~100% | > 80% (with verification) |
| OAuth adoption | 0% | > 30% of new users |
| Support tickets (auth issues) | Baseline | No increase |

---

## Timeline

| Week | Tasks | Deliverables |
|------|-------|--------------|
| Week 1 | Phase 1 (Quotas) | Quota system live, migration applied |
| Week 2 | Phase 2-3 (Email + CAPTCHA) | Verification flow, bot protection |
| Week 3 | Phase 4 (OAuth) | Google + GitHub login |
| Week 4 | Phase 5 (Frontend) | Full UI integration |
| Week 5 | Testing + Rollout | Production deployment |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Email delivery issues | Users can't verify | Multiple SMTP providers, manual verification option |
| OAuth provider outage | Users can't login | Keep password login as fallback |
| Quota too restrictive | User complaints | Start generous, adjust based on feedback |
| CAPTCHA blocks legitimate users | Registration drop | Use invisible CAPTCHA, monitor block rate |

---

## Appendix: API Endpoint Summary

### New Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/auth/google` | Initiate Google OAuth |
| GET | `/api/auth/google/callback` | Google OAuth callback |
| GET | `/api/auth/github` | Initiate GitHub OAuth |
| GET | `/api/auth/github/callback` | GitHub OAuth callback |
| POST | `/api/auth/send-verification` | Send verification email |
| GET | `/api/auth/verify-email/{token}` | Verify email |
| GET | `/api/users/me/quota` | Get current quota status |

### Modified Endpoints

| Endpoint | Change |
|----------|--------|
| `POST /api/auth/register` | Add CAPTCHA validation |
| `POST /api/predictions` | Add quota check |
| `POST /api/sessions/{id}/predictions` | Add quota check |
| `GET /api/users/me` | Include quota info in response |

---

## References

- [Google OAuth Documentation](https://developers.google.com/identity/protocols/oauth2)
- [GitHub OAuth Documentation](https://docs.github.com/en/developers/apps/building-oauth-apps)
- [reCAPTCHA v3 Documentation](https://developers.google.com/recaptcha/docs/v3)
- [Authlib Documentation](https://docs.authlib.org/en/latest/)
