"""
OAuth service for Google and GitHub social login integration.

This service handles:
- OAuth client configuration for Google and GitHub
- Token exchange and user info retrieval
- Account creation and linking for OAuth users
- State management for CSRF protection

Usage:
    from app.services.oauth_service import OAuthService
    
    # Get authorization URL
    url = OAuthService.get_google_authorization_url(state="random_state")
    
    # Exchange code for user info
    user_info = await OAuthService.exchange_google_code(code="auth_code")
"""

import secrets
import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timezone
import uuid

import httpx
from authlib.integrations.httpx_client import AsyncOAuth2Client
from sqlalchemy.orm import Session

from app.config import settings
from app.models.user import User


logger = logging.getLogger("security.oauth")


class OAuthConfig:
    """OAuth provider configuration"""
    
    # Google OAuth endpoints
    GOOGLE_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
    GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
    
    # GitHub OAuth endpoints
    GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
    GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
    GITHUB_USERINFO_URL = "https://api.github.com/user"
    GITHUB_EMAILS_URL = "https://api.github.com/user/emails"


class OAuthService:
    """
    Service for handling OAuth authentication with Google and GitHub.
    
    This service provides methods for:
    - Generating authorization URLs with CSRF protection
    - Exchanging authorization codes for tokens
    - Retrieving user profile information
    - Creating or linking user accounts
    """
    
    # -------------------- Configuration Checks --------------------
    
    @staticmethod
    def is_google_enabled() -> bool:
        """Check if Google OAuth is properly configured"""
        return bool(
            getattr(settings, 'GOOGLE_CLIENT_ID', None) and
            getattr(settings, 'GOOGLE_CLIENT_SECRET', None)
        )
    
    @staticmethod
    def is_github_enabled() -> bool:
        """Check if GitHub OAuth is properly configured"""
        return bool(
            getattr(settings, 'GITHUB_CLIENT_ID', None) and
            getattr(settings, 'GITHUB_CLIENT_SECRET', None)
        )
    
    @staticmethod
    def get_oauth_config() -> Dict[str, Any]:
        """Get OAuth configuration for frontend"""
        return {
            "google": {
                "enabled": OAuthService.is_google_enabled(),
                "client_id": getattr(settings, 'GOOGLE_CLIENT_ID', None) if OAuthService.is_google_enabled() else None
            },
            "github": {
                "enabled": OAuthService.is_github_enabled(),
                "client_id": getattr(settings, 'GITHUB_CLIENT_ID', None) if OAuthService.is_github_enabled() else None
            }
        }
    
    # -------------------- State Management --------------------
    
    @staticmethod
    def generate_state() -> str:
        """Generate a secure random state token for CSRF protection"""
        return secrets.token_urlsafe(32)
    
    # -------------------- Google OAuth --------------------
    
    @staticmethod
    def get_google_authorization_url(state: str, redirect_uri: Optional[str] = None) -> str:
        """
        Generate Google OAuth authorization URL.
        
        Args:
            state: Random state token for CSRF protection
            redirect_uri: Override redirect URI (defaults to settings)
            
        Returns:
            Authorization URL to redirect user to
        """
        if not OAuthService.is_google_enabled():
            raise ValueError("Google OAuth is not configured")
        
        redirect = redirect_uri or getattr(settings, 'GOOGLE_REDIRECT_URI', None) or \
            f"{settings.FRONTEND_URL}/auth/google/callback"
        
        client = AsyncOAuth2Client(
            client_id=settings.GOOGLE_CLIENT_ID,
            client_secret=settings.GOOGLE_CLIENT_SECRET,
            redirect_uri=redirect,
            scope="openid email profile"
        )
        
        url, _ = client.create_authorization_url(
            OAuthConfig.GOOGLE_AUTHORIZE_URL,
            state=state,
            redirect_uri=redirect,
            access_type="offline",
            prompt="select_account"
        )
        
        return url
    
    @staticmethod
    async def exchange_google_code(
        code: str,
        redirect_uri: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Exchange Google authorization code for user info.
        
        Args:
            code: Authorization code from callback
            redirect_uri: Must match the one used in authorization
            
        Returns:
            Tuple of (success, message, user_info)
            user_info contains: id, email, name, picture
        """
        if not OAuthService.is_google_enabled():
            return False, "Google OAuth is not configured", None
        
        redirect = redirect_uri or getattr(settings, 'GOOGLE_REDIRECT_URI',
            f"{settings.FRONTEND_URL}/auth/google/callback")
        
        try:
            async with httpx.AsyncClient() as http_client:
                # Exchange code for token
                token_response = await http_client.post(
                    OAuthConfig.GOOGLE_TOKEN_URL,
                    data={
                        "client_id": settings.GOOGLE_CLIENT_ID,
                        "client_secret": settings.GOOGLE_CLIENT_SECRET,
                        "code": code,
                        "grant_type": "authorization_code",
                        "redirect_uri": redirect
                    },
                    headers={"Accept": "application/json"}
                )
                
                if token_response.status_code != 200:
                    logger.error(f"Google token exchange failed: {token_response.text}")
                    return False, "Failed to exchange authorization code", None
                
                token_data = token_response.json()
                access_token = token_data.get("access_token")
                
                if not access_token:
                    logger.error("No access token in Google response")
                    return False, "No access token received", None
                
                # Get user info
                userinfo_response = await http_client.get(
                    OAuthConfig.GOOGLE_USERINFO_URL,
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                
                if userinfo_response.status_code != 200:
                    logger.error(f"Google userinfo failed: {userinfo_response.text}")
                    return False, "Failed to get user information", None
                
                user_data = userinfo_response.json()
                
                user_info = {
                    "id": user_data.get("id"),
                    "email": user_data.get("email"),
                    "email_verified": user_data.get("verified_email", False),
                    "name": user_data.get("name"),
                    "given_name": user_data.get("given_name"),
                    "family_name": user_data.get("family_name"),
                    "picture": user_data.get("picture"),
                    "provider": "google"
                }
                
                logger.info(f"Google OAuth successful: email={user_info.get('email')}")
                return True, "Successfully authenticated with Google", user_info
                
        except httpx.TimeoutException:
            logger.error("Google OAuth timeout")
            return False, "Authentication timed out", None
        except Exception as e:
            logger.error(f"Google OAuth error: {str(e)}")
            return False, f"Authentication failed: {str(e)}", None
    
    # -------------------- GitHub OAuth --------------------
    
    @staticmethod
    def get_github_authorization_url(state: str, redirect_uri: Optional[str] = None) -> str:
        """
        Generate GitHub OAuth authorization URL.
        
        Args:
            state: Random state token for CSRF protection
            redirect_uri: Override redirect URI (defaults to settings)
            
        Returns:
            Authorization URL to redirect user to
        """
        if not OAuthService.is_github_enabled():
            raise ValueError("GitHub OAuth is not configured")
        
        redirect = redirect_uri or getattr(settings, 'GITHUB_REDIRECT_URI', None) or \
            f"{settings.FRONTEND_URL}/auth/github/callback"
        
        client = AsyncOAuth2Client(
            client_id=settings.GITHUB_CLIENT_ID,
            client_secret=settings.GITHUB_CLIENT_SECRET,
            redirect_uri=redirect,
            scope="user:email read:user"
        )
        
        url, _ = client.create_authorization_url(
            OAuthConfig.GITHUB_AUTHORIZE_URL,
            state=state,
            redirect_uri=redirect
        )
        
        return url
    
    @staticmethod
    async def exchange_github_code(
        code: str,
        redirect_uri: Optional[str] = None
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Exchange GitHub authorization code for user info.
        
        Args:
            code: Authorization code from callback
            redirect_uri: Must match the one used in authorization
            
        Returns:
            Tuple of (success, message, user_info)
            user_info contains: id, email, name, login (username), avatar_url
        """
        if not OAuthService.is_github_enabled():
            return False, "GitHub OAuth is not configured", None
        
        redirect = redirect_uri or getattr(settings, 'GITHUB_REDIRECT_URI',
            f"{settings.FRONTEND_URL}/auth/github/callback")
        
        try:
            async with httpx.AsyncClient() as http_client:
                # Exchange code for token
                token_response = await http_client.post(
                    OAuthConfig.GITHUB_TOKEN_URL,
                    data={
                        "client_id": settings.GITHUB_CLIENT_ID,
                        "client_secret": settings.GITHUB_CLIENT_SECRET,
                        "code": code,
                        "redirect_uri": redirect
                    },
                    headers={"Accept": "application/json"}
                )
                
                if token_response.status_code != 200:
                    logger.error(f"GitHub token exchange failed: {token_response.text}")
                    return False, "Failed to exchange authorization code", None
                
                token_data = token_response.json()
                access_token = token_data.get("access_token")
                
                if not access_token:
                    error = token_data.get("error_description", "Unknown error")
                    logger.error(f"GitHub token error: {error}")
                    return False, f"Authentication failed: {error}", None
                
                # Get user info
                userinfo_response = await http_client.get(
                    OAuthConfig.GITHUB_USERINFO_URL,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Accept": "application/vnd.github+json"
                    }
                )
                
                if userinfo_response.status_code != 200:
                    logger.error(f"GitHub userinfo failed: {userinfo_response.text}")
                    return False, "Failed to get user information", None
                
                user_data = userinfo_response.json()
                
                # GitHub doesn't always return email in user info, need to fetch separately
                email = user_data.get("email")
                email_verified = False
                
                if not email:
                    emails_response = await http_client.get(
                        OAuthConfig.GITHUB_EMAILS_URL,
                        headers={
                            "Authorization": f"Bearer {access_token}",
                            "Accept": "application/vnd.github+json"
                        }
                    )
                    
                    if emails_response.status_code == 200:
                        emails = emails_response.json()
                        # Find primary verified email
                        for email_entry in emails:
                            if email_entry.get("primary") and email_entry.get("verified"):
                                email = email_entry.get("email")
                                email_verified = True
                                break
                        # Fallback to any verified email
                        if not email:
                            for email_entry in emails:
                                if email_entry.get("verified"):
                                    email = email_entry.get("email")
                                    email_verified = True
                                    break
                
                user_info = {
                    "id": str(user_data.get("id")),
                    "email": email,
                    "email_verified": email_verified,
                    "name": user_data.get("name") or user_data.get("login"),
                    "login": user_data.get("login"),
                    "avatar_url": user_data.get("avatar_url"),
                    "provider": "github"
                }
                
                logger.info(f"GitHub OAuth successful: login={user_info.get('login')}")
                return True, "Successfully authenticated with GitHub", user_info
                
        except httpx.TimeoutException:
            logger.error("GitHub OAuth timeout")
            return False, "Authentication timed out", None
        except Exception as e:
            logger.error(f"GitHub OAuth error: {str(e)}")
            return False, f"Authentication failed: {str(e)}", None
    
    # -------------------- User Account Management --------------------
    
    @staticmethod
    def find_user_by_oauth(
        db: Session,
        provider: str,
        oauth_id: str
    ) -> Optional[User]:
        """
        Find user by OAuth provider and ID.
        
        Args:
            db: Database session
            provider: OAuth provider ('google' or 'github')
            oauth_id: Provider's user ID
            
        Returns:
            User if found, None otherwise
        """
        if provider == "google":
            return db.query(User).filter(User.google_id == oauth_id).first()
        elif provider == "github":
            return db.query(User).filter(User.github_id == oauth_id).first()
        return None
    
    @staticmethod
    def find_user_by_email(db: Session, email: str) -> Optional[User]:
        """Find user by email address"""
        if not email:
            return None
        return db.query(User).filter(User.email == email).first()
    
    @staticmethod
    def create_oauth_user(
        db: Session,
        provider: str,
        oauth_id: str,
        email: Optional[str],
        name: Optional[str],
        username_hint: Optional[str] = None,
        email_verified: bool = False
    ) -> Tuple[bool, str, Optional[User]]:
        """
        Create a new user from OAuth login.
        
        Args:
            db: Database session
            provider: OAuth provider ('google' or 'github')
            oauth_id: Provider's user ID
            email: User's email
            name: User's display name
            username_hint: Suggested username (e.g., GitHub login)
            email_verified: Whether the email is verified by the provider
            
        Returns:
            Tuple of (success, message, user)
        """
        try:
            # Generate unique username
            base_username = username_hint or (email.split("@")[0] if email else f"{provider}_user")
            username = OAuthService._generate_unique_username(db, base_username)
            
            # Generate UUID for key_id
            key_id = str(uuid.uuid4())
            
            # Create user - no password required for OAuth users
            new_user = User(
                key_id=key_id,
                username=username,
                email=email,
                password_hash=None,  # OAuth users don't have password
                role="user",
                is_active=True,
                email_verified=email_verified,
                oauth_provider=provider,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            # Set provider-specific ID
            if provider == "google":
                new_user.google_id = oauth_id
            elif provider == "github":
                new_user.github_id = oauth_id
            
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            
            logger.info(f"OAuth user created: username={username}, provider={provider}")
            return True, "User created successfully", new_user
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create OAuth user: {str(e)}")
            return False, f"Failed to create user: {str(e)}", None
    
    @staticmethod
    def link_oauth_account(
        db: Session,
        user: User,
        provider: str,
        oauth_id: str,
        email: Optional[str] = None,
        email_verified: bool = False
    ) -> Tuple[bool, str]:
        """
        Link an OAuth account to an existing user.
        
        Args:
            db: Database session
            user: Existing user to link to
            provider: OAuth provider ('google' or 'github')
            oauth_id: Provider's user ID
            email: Email from provider (used to verify/set email)
            email_verified: Whether email is verified by provider
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Check if OAuth account is already linked to another user
            existing = OAuthService.find_user_by_oauth(db, provider, oauth_id)
            if existing and existing.key_id != user.key_id:
                return False, f"This {provider} account is already linked to another user"
            
            # Link the account
            if provider == "google":
                user.google_id = oauth_id
            elif provider == "github":
                user.github_id = oauth_id
            
            # Update email if not set and provider email is verified
            if not user.email and email and email_verified:
                # Check email isn't used by another account
                email_user = OAuthService.find_user_by_email(db, email)
                if not email_user:
                    user.email = email
                    user.email_verified = True
            
            user.updated_at = datetime.now(timezone.utc)
            db.commit()
            
            logger.info(f"OAuth account linked: user={user.username}, provider={provider}")
            return True, f"{provider.title()} account linked successfully"
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to link OAuth account: {str(e)}")
            return False, f"Failed to link account: {str(e)}"
    
    @staticmethod
    def unlink_oauth_account(
        db: Session,
        user: User,
        provider: str
    ) -> Tuple[bool, str]:
        """
        Unlink an OAuth account from a user.
        
        Args:
            db: Database session
            user: User to unlink from
            provider: OAuth provider ('google' or 'github')
            
        Returns:
            Tuple of (success, message)
        """
        try:
            # Check user has password or another OAuth method
            has_password = user.password_hash is not None
            has_google = user.google_id is not None and provider != "google"
            has_github = user.github_id is not None and provider != "github"
            
            if not (has_password or has_google or has_github):
                return False, "Cannot unlink the only authentication method. Please set a password first."
            
            # Unlink the account
            if provider == "google":
                user.google_id = None
            elif provider == "github":
                user.github_id = None
            
            # Clear oauth_provider if no providers linked
            if not user.google_id and not user.github_id:
                user.oauth_provider = None
            
            user.updated_at = datetime.now(timezone.utc)
            db.commit()
            
            logger.info(f"OAuth account unlinked: user={user.username}, provider={provider}")
            return True, f"{provider.title()} account unlinked successfully"
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to unlink OAuth account: {str(e)}")
            return False, f"Failed to unlink account: {str(e)}"
    
    @staticmethod
    def _generate_unique_username(db: Session, base: str) -> str:
        """Generate a unique username from a base name"""
        # Clean the base username
        import re
        clean_base = re.sub(r'[^a-zA-Z0-9_-]', '', base)[:40]
        if not clean_base:
            clean_base = "user"
        
        # Check if base is available
        if not db.query(User).filter(User.username == clean_base).first():
            return clean_base
        
        # Try adding numbers
        for i in range(1, 10000):
            candidate = f"{clean_base}_{i}"
            if not db.query(User).filter(User.username == candidate).first():
                return candidate
        
        # Fallback to UUID
        return f"{clean_base}_{uuid.uuid4().hex[:8]}"
    
    @staticmethod
    def authenticate_or_create_oauth_user(
        db: Session,
        provider: str,
        user_info: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[User], bool]:
        """
        Authenticate or create a user from OAuth login.
        
        This is the main entry point for OAuth login flow:
        1. Check if user exists by OAuth ID
        2. If not, check by email and link account
        3. If no match, create new user
        
        Args:
            db: Database session
            provider: OAuth provider ('google' or 'github')
            user_info: User info from OAuth provider
            
        Returns:
            Tuple of (success, message, user, is_new_user)
        """
        oauth_id = user_info.get("id")
        email = user_info.get("email")
        name = user_info.get("name")
        email_verified = user_info.get("email_verified", False)
        username_hint = user_info.get("login")  # GitHub login
        
        if not oauth_id:
            return False, "No user ID from OAuth provider", None, False
        
        # 1. Check if user exists by OAuth ID
        user = OAuthService.find_user_by_oauth(db, provider, oauth_id)
        if user:
            # Update last login
            user.last_login = datetime.now(timezone.utc)
            db.commit()
            logger.info(f"OAuth login: existing user={user.username}, provider={provider}")
            return True, "Login successful", user, False
        
        # 2. Check if user exists by email
        if email:
            user = OAuthService.find_user_by_email(db, email)
            if user:
                # Link OAuth account to existing user
                success, message = OAuthService.link_oauth_account(
                    db, user, provider, oauth_id, email, email_verified
                )
                if success:
                    user.last_login = datetime.now(timezone.utc)
                    db.commit()
                    logger.info(f"OAuth account linked on login: user={user.username}, provider={provider}")
                    return True, "Login successful (account linked)", user, False
                else:
                    return False, message, None, False
        
        # 3. Create new user
        success, message, user = OAuthService.create_oauth_user(
            db=db,
            provider=provider,
            oauth_id=oauth_id,
            email=email,
            name=name,
            username_hint=username_hint,
            email_verified=email_verified
        )
        
        if success and user:
            user.last_login = datetime.now(timezone.utc)
            db.commit()
            return True, "Account created successfully", user, True
        
        return False, message, None, False
