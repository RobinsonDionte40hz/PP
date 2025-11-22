# Authentication Flow Diagrams

This document provides visual representations of the authentication flows implemented in the Protein Prediction Platform.

## Table of Contents

- [Registration Flow](#registration-flow)
- [Login Flow](#login-flow)
- [Token Refresh Flow](#token-refresh-flow)
- [Logout Flow](#logout-flow)
- [Protected Resource Access](#protected-resource-access)
- [Session Management](#session-management)
- [Error Handling Flows](#error-handling-flows)

---

## Registration Flow

### Sequence Diagram

```
User                Frontend            Backend API         Database          Redis
 |                     |                    |                   |               |
 |--Fill Form--------->|                    |                   |               |
 |                     |                    |                   |               |
 |--Submit------------>|                    |                   |               |
 |                     |                    |                   |               |
 |                     |--POST /auth/-------->|                   |               |
 |                     |    register       |                   |               |
 |                     |                    |                   |               |
 |                     |                    |--Check Rate------>|               |
 |                     |                    |    Limit          |               |
 |                     |                    |<--(Allow/Deny)---|               |
 |                     |                    |                   |               |
 |                     |                    |--Validate Input   |               |
 |                     |                    |                   |               |
 |                     |                    |--Check Unique---->|               |
 |                     |                    |    Username       |               |
 |                     |                    |<--Result----------|               |
 |                     |                    |                   |               |
 |                     |                    |--Hash Password    |               |
 |                     |                    |  (bcrypt)         |               |
 |                     |                    |                   |               |
 |                     |                    |--Create User----->|               |
 |                     |                    |    Record         |               |
 |                     |                    |<--User Created----|               |
 |                     |                    |                   |               |
 |                     |<--201 Created------|                   |               |
 |                     |  {user data}       |                   |               |
 |                     |                    |                   |               |
 |<--Success Message---|                    |                   |               |
 |   (Toast)           |                    |                   |               |
 |                     |                    |                   |               |
 |                     |--Auto Login------->|                   |               |
 |                     |  (see Login Flow)  |                   |               |
```

### Steps

1. **User fills registration form**
   - Username (3-50 characters)
   - Email (optional)
   - Password (8-72 characters with complexity requirements)
   - Confirm password

2. **Frontend validation**
   - Check password match
   - Validate email format
   - Check password complexity

3. **Backend receives request**
   - Rate limit check (5 registrations/hour per IP)
   - Validate input format
   - Check username uniqueness
   - Check email uniqueness (if provided)

4. **Password processing**
   - Hash password with bcrypt (cost factor 12)
   - Generate UUID for user key_id

5. **Database operations**
   - Insert user record
   - Store username, email, password_hash, created_at

6. **Response**
   - Return user profile (without password)
   - Show success notification
   - Auto-login user (optional)

### Error Scenarios

- **Rate Limit Exceeded**: 429 response with retry-after header
- **Username Exists**: 409 Conflict
- **Email Exists**: 409 Conflict
- **Weak Password**: 400 Bad Request with specific error
- **Invalid Format**: 400 Bad Request with validation errors

---

## Login Flow

### Sequence Diagram

```
User                Frontend            Backend API         Database          Redis
 |                     |                    |                   |               |
 |--Enter Credentials->|                    |                   |               |
 |                     |                    |                   |               |
 |--Click Login------->|                    |                   |               |
 |                     |                    |                   |               |
 |                     |--POST /auth/-------->|                   |               |
 |                     |    login          |                   |               |
 |                     |  {username, pwd}  |                   |               |
 |                     |                    |                   |               |
 |                     |                    |--Check Rate------>|               |
 |                     |                    |    Limit          |               |
 |                     |                    |<--(Allow/Deny)---|               |
 |                     |                    |                   |               |
 |                     |                    |--Get User-------->|               |
 |                     |                    |    by Username    |               |
 |                     |                    |<--User Record-----|               |
 |                     |                    |                   |               |
 |                     |                    |--Verify Password  |               |
 |                     |                    |  (bcrypt.verify)  |               |
 |                     |                    |                   |               |
 |                     |                    |--Check Active-------------------->|
 |                     |                    |    Session        |               |
 |                     |                    |<--Session ID---------------------|
 |                     |                    |                   |               |
 |                     |                    |--Delete Old--------------------->|
 |                     |                    |    Session        |               |
 |                     |                    |<--Deleted------------------------|
 |                     |                    |                   |               |
 |                     |                    |--Generate Tokens  |               |
 |                     |                    |  (access+refresh) |               |
 |                     |                    |                   |               |
 |                     |                    |--Create Session------------------>|
 |                     |                    |  (30min TTL)      |               |
 |                     |                    |<--Session Created----------------|
 |                     |                    |                   |               |
 |                     |                    |--Store User->-------------------->|
 |                     |                    |    Session Map    |               |
 |                     |                    |<--Stored-------------------------|
 |                     |                    |                   |               |
 |                     |<--200 OK-----------|                   |               |
 |                     |  {user, tokens}    |                   |               |
 |                     |                    |                   |               |
 |<--Store Tokens------|                    |                   |               |
 |   localStorage      |                    |                   |               |
 |                     |                    |                   |               |
 |<--Success Message---|                    |                   |               |
 |   (Toast)           |                    |                   |               |
 |                     |                    |                   |               |
 |<--Redirect----------|                    |                   |               |
 |   to Dashboard      |                    |                   |               |
```

### Steps

1. **User enters credentials**
   - Username
   - Password

2. **Frontend validation**
   - Check non-empty fields
   - Show loading state

3. **Backend receives request**
   - Rate limit check (10 logins/15min per IP)
   - Validate credentials presence

4. **Authentication**
   - Query user by username
   - Verify password hash with bcrypt
   - Check if user exists

5. **Session management**
   - Check for existing active session
   - Terminate old session if exists (single-session enforcement)
   - Generate new JWT tokens
     - Access token (30 min expiry)
     - Refresh token (7 day expiry)

6. **Redis operations**
   - Create session with user data
   - Set TTL (30 minutes)
   - Map user_id -> session_id for single-session tracking

7. **Response**
   - Return user profile + tokens
   - Frontend stores tokens
   - Redirect to dashboard

### Error Scenarios

- **Invalid Credentials**: 401 Unauthorized
- **Rate Limit**: 429 Too Many Requests
- **User Not Found**: 401 Unauthorized (same message as invalid password)
- **Session Creation Failed**: 500 Internal Server Error

---

## Token Refresh Flow

### Sequence Diagram

```
Frontend            Backend API         Redis
   |                    |                 |
   |--Token Expiring--->|                 |
   |  (5 min before)    |                 |
   |                    |                 |
   |--POST /auth/-------->|                 |
   |    refresh        |                 |
   |  {refresh_token}  |                 |
   |                    |                 |
   |                    |--Verify Token   |
   |                    |  (JWT)          |
   |                    |                 |
   |                    |--Get Session--->|
   |                    |  by JTI         |
   |                    |<--Session------|
   |                    |                 |
   |                    |--Check Valid    |
   |                    |  Session        |
   |                    |                 |
   |                    |--Generate New   |
   |                    |  Access Token   |
   |                    |                 |
   |                    |--Update-------->|
   |                    |  Session TTL    |
   |                    |<--Updated------|
   |                    |                 |
   |<--200 OK-----------|                 |
   |  {access_token}    |                 |
   |                    |                 |
   |--Update Token----->|                 |
   |  in Storage        |                 |
```

### Steps

1. **Token expiration detection**
   - Frontend monitors token expiry
   - Automatically triggers refresh 5 minutes before expiration
   - Or manually triggered on 401 response

2. **Refresh request**
   - Send refresh token to backend
   - Refresh tokens valid for 7 days

3. **Backend validation**
   - Verify refresh token signature
   - Check token type (must be "refresh")
   - Extract JTI (JWT ID) from token
   - Verify session still exists in Redis

4. **Token generation**
   - Generate new access token (30 min expiry)
   - Keep existing refresh token
   - Update session expiration in Redis

5. **Response**
   - Return new access token
   - Frontend replaces old token

### Error Scenarios

- **Invalid Token**: 401 Unauthorized
- **Expired Refresh Token**: 401 Unauthorized (user must login again)
- **Session Not Found**: 401 Unauthorized (session was terminated)
- **Wrong Token Type**: 401 Unauthorized

---

## Logout Flow

### Sequence Diagram

```
User              Frontend            Backend API         Redis
 |                   |                    |                 |
 |--Click Logout---->|                    |                 |
 |                   |                    |                 |
 |                   |--POST /auth/-------->|                 |
 |                   |    logout         |                 |
 |                   |  (Bearer token)   |                 |
 |                   |                    |                 |
 |                   |                    |--Verify Token   |
 |                   |                    |                 |
 |                   |                    |--Extract JTI    |
 |                   |                    |                 |
 |                   |                    |--Delete-------->|
 |                   |                    |  Session        |
 |                   |                    |<--Deleted------|
 |                   |                    |                 |
 |                   |                    |--Remove-------->|
 |                   |                    |  User->Session  |
 |                   |                    |  Mapping        |
 |                   |                    |<--Removed------|
 |                   |                    |                 |
 |                   |<--200 OK-----------|                 |
 |                   |  {message}         |                 |
 |                   |                    |                 |
 |<--Clear Tokens----|                    |                 |
 |   from Storage    |                    |                 |
 |                   |                    |                 |
 |<--Success---------|                    |                 |
 |   (Toast)         |                    |                 |
 |                   |                    |                 |
 |<--Redirect--------|                    |                 |
 |   to Login        |                    |                 |
```

### Steps

1. **User initiates logout**
   - Click logout button
   - Or automatic logout on error

2. **Frontend sends request**
   - Include access token in Authorization header
   - Show loading state

3. **Backend processes**
   - Verify access token
   - Extract session ID (JTI) from token
   - Delete session from Redis
   - Remove user->session mapping

4. **Frontend cleanup**
   - Clear tokens from storage
   - Clear user data
   - Reset auth state
   - Redirect to login page

### Error Scenarios

- **Invalid Token**: 401 Unauthorized (still clear local tokens)
- **Session Not Found**: Continue with local cleanup
- **Redis Error**: Log error, continue with response

---

## Protected Resource Access

### Sequence Diagram

```
Frontend            Backend API         Redis
   |                    |                 |
   |--GET /api/-------->|                 |
   |  predictions       |                 |
   |  (Bearer token)    |                 |
   |                    |                 |
   |              [Auth Middleware]       |
   |                    |                 |
   |                    |--Verify Token   |
   |                    |  (JWT)          |
   |                    |                 |
   |                    |--Extract JTI    |
   |                    |                 |
   |                    |--Get Session--->|
   |                    |<--Session------|
   |                    |                 |
   |                    |--Check Match    |
   |                    |  Token JTI ==   |
   |                    |  Session JTI    |
   |                    |                 |
   |                    |--Attach User    |
   |                    |  to Request     |
   |                    |                 |
   |              [Route Handler]         |
   |                    |                 |
   |                    |--Process        |
   |                    |  Request        |
   |                    |                 |
   |<--200 OK-----------|                 |
   |  {data}            |                 |
```

### Steps

1. **Frontend makes request**
   - Include Authorization header
   - Format: `Bearer <access_token>`

2. **Authentication middleware**
   - Extract token from header
   - Verify JWT signature
   - Check token expiration
   - Extract claims (sub, username, jti)

3. **Session validation**
   - Get session from Redis using JTI
   - Verify session exists
   - Check single-session constraint (token JTI matches active session)

4. **Request processing**
   - Attach user info to request context
   - Pass to route handler
   - Handler has access to current user

5. **Response**
   - Return requested data
   - Session TTL auto-extends on activity

### Error Scenarios

- **Missing Token**: 401 Unauthorized
- **Invalid Token**: 401 Unauthorized
- **Expired Token**: 401 Unauthorized (trigger refresh)
- **Session Mismatch**: 401 Unauthorized (user logged in elsewhere)
- **Session Expired**: 401 Unauthorized

---

## Session Management

### Session Lifecycle

```
Login                  Activity                 Inactivity              Logout
  |                       |                         |                      |
  |--Create Session------>|                         |                      |
  |  (30min TTL)          |                         |                      |
  |                       |                         |                      |
  |                       |--API Request----------->|                      |
  |                       |  (extends TTL)          |                      |
  |                       |                         |                      |
  |                       |                    [30min pass]               |
  |                       |                         |                      |
  |                       |                         |--Auto Expire-------->|
  |                       |                         |                      |
  |                       |                         |                      |
  |                       |--Manual Logout----------------------->|        |
  |                       |                                       |        |
  |                       |                                   [Delete]     |
```

### Session Data Structure (Redis)

```json
{
  "session:{jti}": {
    "user_id": "550e8400-e29b-41d4-a716-446655440000",
    "username": "john_doe",
    "jti": "unique-jwt-id",
    "ip_address": "192.168.1.100",
    "user_agent": "Mozilla/5.0...",
    "created_at": "2025-11-22T10:30:00",
    "last_activity": "2025-11-22T10:45:00"
  },
  "TTL": 1800  // 30 minutes in seconds
}
```

```json
{
  "user_session:{user_id}": "session:{jti}",
  "TTL": 1800
}
```

### Single-Session Enforcement

```
User Login (Device A)     User Login (Device B)     Device A Access
        |                         |                        |
        |--Create Session A------>|                        |
        |   in Redis              |                        |
        |                         |                        |
        |                         |--Create Session B----->|
        |                         |   (replaces A)         |
        |                         |                        |
        |                         |--Delete Session A----->|
        |                         |                        |
        |                         |                        |
        |                                     |--Request-->|
        |                                     |  (token A) |
        |                                     |            |
        |                                     |<--401------|
        |                                     | Session    |
        |                                     | Mismatch   |
```

When a user logs in:
1. Check for existing active session
2. If found, delete old session from Redis
3. Create new session
4. Old tokens become invalid (session mismatch)

---

## Error Handling Flows

### Network Error with Retry

```
Frontend                  Backend API
   |                          |
   |--API Request------------>|
   |                          X (Network Error)
   |                          
   |--Retry Attempt 1 (1s)--->|
   |                          X (Network Error)
   |                          
   |--Retry Attempt 2 (2s)--->|
   |                          |
   |<--200 OK----------------|
```

### Token Expiration with Auto-Refresh

```
Frontend                  Backend API
   |                          |
   |--API Request------------>|
   |  (expired token)         |
   |                          |
   |<--401 Unauthorized------|
   |                          |
   |--POST /auth/refresh----->|
   |                          |
   |<--200 OK----------------|
   |  {new access_token}      |
   |                          |
   |--Retry Original--------->|
   |  (new token)             |
   |                          |
   |<--200 OK----------------|
   |  {data}                  |
```

### Rate Limit Error

```
User              Frontend            Backend API         Redis
 |                   |                    |                 |
 |--Multiple-------->|                    |                 |
 |  Login Attempts   |                    |                 |
 |                   |                    |                 |
 |                   |--POST /auth/-------->|                 |
 |                   |    login (11th)   |                 |
 |                   |                    |                 |
 |                   |                    |--Check Rate---->|
 |                   |                    |<--(Exceeded)---|
 |                   |                    |                 |
 |                   |<--429 Error--------|                 |
 |                   |  {retry_after: 60} |                 |
 |                   |                    |                 |
 |<--Error Message---|                    |                 |
 |  "Try again in 60s"|                   |                 |
 |                   |                    |                 |
 |                   [Wait 60 seconds]    |                 |
 |                   |                    |                 |
 |--Retry----------->|--POST /auth/-------->|                 |
 |                   |    login          |                 |
 |                   |                    |<--(Allowed)----|
 |                   |<--200 OK-----------|                 |
```

---

## Security Flow Diagrams

### Password Hashing Flow

```
Registration               Bcrypt               Database
     |                        |                    |
     |--Plain Password------->|                    |
     |                        |                    |
     |                        |--Generate Salt     |
     |                        |  (cost=12)         |
     |                        |                    |
     |                        |--Hash Password     |
     |                        |  (1-2 seconds)     |
     |                        |                    |
     |<--Hash-----------------|                    |
     |                        |                    |
     |--Store Hash-------------------------------->|
     |                        |                    |

Login                     Bcrypt               Database
     |                        |                    |
     |--Plain Password------->|                    |
     |                        |                    |
     |                        |<--Get Hash---------|
     |                        |                    |
     |                        |--Compare           |
     |                        |  (constant-time)   |
     |                        |                    |
     |<--Match/No Match------|                    |
```

### CSRF Protection Flow

```
Frontend            Backend API
   |                    |
   |--GET Request------>|
   |                    |
   |<--Response---------|
   |  (Set-Cookie:      |
   |   csrf_token)      |
   |                    |
   |--POST Request----->|
   |  (Cookie: csrf)    |
   |  (Header: csrf)    |
   |                    |
   |                    |--Verify Match
   |                    |  Cookie == Header
   |                    |
   |<--200 OK-----------|
```

---

## Implementation Notes

### JWT Token Structure

**Access Token Claims**:
```json
{
  "sub": "550e8400-e29b-41d4-a716-446655440000",  // User ID
  "username": "john_doe",                         // Username
  "jti": "unique-jwt-id",                         // Token ID
  "type": "access",                               // Token type
  "exp": 1700654400,                             // Expiration (Unix timestamp)
  "iat": 1700652600                              // Issued at
}
```

**Refresh Token Claims**:
```json
{
  "sub": "550e8400-e29b-41d4-a716-446655440000",
  "jti": "unique-jwt-id",
  "type": "refresh",
  "exp": 1701259200,  // 7 days later
  "iat": 1700652600
}
```

### Rate Limiting Configuration

| Endpoint | Limit | Window | Key |
|----------|-------|--------|-----|
| `/auth/register` | 5 | 1 hour | IP address |
| `/auth/login` | 10 | 15 minutes | IP address |
| `/auth/refresh` | 20 | 1 hour | User ID |

### Session Expiration

- **Initial TTL**: 30 minutes
- **Auto-Extend**: On each API request
- **Max Session Length**: 7 days (refresh token expiry)
- **Idle Timeout**: 30 minutes of inactivity

---

## Troubleshooting Guide

### Common Issues

**Issue**: 401 Unauthorized on valid token
- **Cause**: Session mismatch (logged in elsewhere)
- **Solution**: User must login again

**Issue**: Token refresh fails
- **Cause**: Refresh token expired (7 days)
- **Solution**: User must login again

**Issue**: 429 Too Many Requests
- **Cause**: Rate limit exceeded
- **Solution**: Wait for retry-after period

**Issue**: Session not found
- **Cause**: Redis restarted or session expired
- **Solution**: User must login again

**Issue**: CSRF token mismatch
- **Cause**: Cookie and header don't match
- **Solution**: Refresh page to get new CSRF token

---

## Related Documentation

- [API Documentation](./API.md) - Detailed API endpoint reference
- [Error Handling](./ERROR_HANDLING.md) - Frontend error handling system
- [Security Guide](../backend/SECURITY.md) - Backend security implementation
- [Setup Guide](./SETUP.md) - Installation and configuration

---

**Last Updated**: November 22, 2025
**Version**: 1.0.0
