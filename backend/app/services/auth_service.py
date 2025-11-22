"""
Authentication service for user registration and management
"""
import uuid
from datetime import datetime, timezone
from typing import Tuple, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.models.user import User
from app.utils.password import hash_password, validate_password_strength, validate_credentials


class AuthService:
    """Service for handling user authentication operations"""
    
    @staticmethod
    def register_user(
        db: Session,
        username: str,
        password: str,
        email: Optional[str] = None
    ) -> Tuple[bool, str, Optional[User]]:
        """
        Register a new user with validation.
        
        Args:
            db: Database session
            username: Username for the new user
            password: Plain text password
            email: Optional email address
            
        Returns:
            Tuple of (success: bool, message: str, user: Optional[User])
            
        Example:
            >>> success, message, user = AuthService.register_user(db, "john_doe", "Pass123!", "john@example.com")
            >>> if success:
            ...     print(f"User created: {user.username}")
        """
        # Validate credentials are not empty
        valid, errors = validate_credentials(username, password)
        if not valid:
            return False, "; ".join(errors), None
        
        # Validate password strength
        valid, errors = validate_password_strength(password)
        if not valid:
            return False, "; ".join(errors), None
        
        # Check if username already exists
        existing_user = db.query(User).filter(User.username == username).first()
        if existing_user:
            return False, "Username already exists", None
        
        # Check if email already exists (if provided)
        if email:
            existing_email = db.query(User).filter(User.email == email).first()
            if existing_email:
                return False, "Email already exists", None
        
        try:
            # Generate UUID for key_id
            key_id = str(uuid.uuid4())
            
            # Hash password
            password_hash = hash_password(password)
            
            # Create user
            new_user = User(
                key_id=key_id,
                username=username,
                email=email,
                password_hash=password_hash,
                is_active=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            
            return True, "User registered successfully", new_user
            
        except IntegrityError as e:
            db.rollback()
            # Handle race condition where username was taken between check and insert
            if "username" in str(e.orig).lower():
                return False, "Username already exists", None
            elif "email" in str(e.orig).lower():
                return False, "Email already exists", None
            else:
                return False, f"Database error: {str(e.orig)}", None
        except Exception as e:
            db.rollback()
            return False, f"Registration failed: {str(e)}", None
    
    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        """
        Get user by username.
        
        Args:
            db: Database session
            username: Username to search for
            
        Returns:
            User object if found, None otherwise
        """
        return db.query(User).filter(User.username == username).first()
    
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """
        Get user by email.
        
        Args:
            db: Database session
            email: Email to search for
            
        Returns:
            User object if found, None otherwise
        """
        return db.query(User).filter(User.email == email).first()
    
    @staticmethod
    def get_user_by_id(db: Session, key_id: str) -> Optional[User]:
        """
        Get user by key_id.
        
        Args:
            db: Database session
            key_id: User UUID to search for
            
        Returns:
            User object if found, None otherwise
        """
        return db.query(User).filter(User.key_id == key_id).first()
