"""
Email Service

This service handles sending emails via SMTP.
Supports:
- Verification emails
- Password reset emails
- General notifications
"""
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


class EmailService:
    """Service for sending emails via SMTP"""
    
    def __init__(self):
        self.smtp_host = settings.SMTP_HOST
        self.smtp_port = settings.SMTP_PORT
        self.smtp_user = settings.SMTP_USER
        self.smtp_password = settings.SMTP_PASSWORD
        self.from_email = settings.SMTP_USER or "noreply@emergentfolds.com"
    
    def _is_configured(self) -> bool:
        """Check if SMTP is configured"""
        return bool(self.smtp_host and self.smtp_user and self.smtp_password)
    
    def _send_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None
    ) -> bool:
        """
        Send an email.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            html_body: HTML content
            text_body: Plain text content (optional, derived from html if not provided)
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self._is_configured():
            logger.warning("SMTP not configured. Email not sent.")
            # In development, log the email instead
            if settings.APP_ENV == "development":
                logger.info(f"[DEV] Email would be sent to {to_email}")
                logger.info(f"[DEV] Subject: {subject}")
                logger.info(f"[DEV] Body (HTML): {html_body[:500]}...")
                return True  # Return True in dev to not block functionality
            return False
        
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["From"] = self.from_email
            msg["To"] = to_email
            msg["Subject"] = subject
            
            # Add plain text version
            if text_body:
                msg.attach(MIMEText(text_body, "plain"))
            
            # Add HTML version
            msg.attach(MIMEText(html_body, "html"))
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.from_email, to_email, msg.as_string())
            
            logger.info(f"Email sent to {to_email}: {subject}")
            return True
            
        except smtplib.SMTPAuthenticationError:
            logger.error("SMTP authentication failed")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {e}")
            return False
        except Exception as e:
            logger.exception(f"Failed to send email: {e}")
            return False
    
    def send_verification_email(
        self,
        to_email: str,
        username: str,
        verification_url: str,
        expire_hours: int = 24
    ) -> bool:
        """
        Send email verification email.
        
        Args:
            to_email: User's email address
            username: User's username
            verification_url: URL for verification
            expire_hours: Hours until link expires
            
        Returns:
            True if sent successfully
        """
        subject = "Verify your EmergentFolds account"
        
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 8px 8px 0 0; text-align: center;">
        <h1 style="color: white; margin: 0; font-size: 28px;">EmergentFolds</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0;">Protein Structure Prediction</p>
    </div>
    
    <div style="background: #ffffff; padding: 30px; border: 1px solid #e0e0e0; border-top: none; border-radius: 0 0 8px 8px;">
        <h2 style="color: #333; margin-top: 0;">Verify Your Email Address</h2>
        
        <p>Hi <strong>{username}</strong>,</p>
        
        <p>Welcome to EmergentFolds! To start making protein structure predictions, please verify your email address by clicking the button below:</p>
        
        <div style="text-align: center; margin: 30px 0;">
            <a href="{verification_url}" 
               style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                      color: white; 
                      padding: 14px 32px; 
                      text-decoration: none; 
                      border-radius: 6px; 
                      font-weight: bold;
                      display: inline-block;">
                Verify Email Address
            </a>
        </div>
        
        <p style="color: #666; font-size: 14px;">
            This link will expire in <strong>{expire_hours} hours</strong>.
        </p>
        
        <p style="color: #666; font-size: 14px;">
            If you didn't create an account on EmergentFolds, you can safely ignore this email.
        </p>
        
        <hr style="border: none; border-top: 1px solid #e0e0e0; margin: 30px 0;">
        
        <p style="color: #999; font-size: 12px; margin-bottom: 0;">
            If the button doesn't work, copy and paste this link into your browser:
        </p>
        <p style="color: #667eea; font-size: 12px; word-break: break-all; margin-top: 5px;">
            {verification_url}
        </p>
    </div>
    
    <div style="text-align: center; padding: 20px; color: #999; font-size: 12px;">
        <p style="margin: 0;">© 2025 EmergentFolds. All rights reserved.</p>
        <p style="margin: 5px 0 0 0;">
            <a href="https://emergentfolds.com" style="color: #667eea; text-decoration: none;">emergentfolds.com</a>
        </p>
    </div>
</body>
</html>
"""
        
        text_body = f"""
EmergentFolds - Verify Your Email Address

Hi {username},

Welcome to EmergentFolds! To start making protein structure predictions, please verify your email address by visiting the link below:

{verification_url}

This link will expire in {expire_hours} hours.

If you didn't create an account on EmergentFolds, you can safely ignore this email.

---
© 2025 EmergentFolds. All rights reserved.
https://emergentfolds.com
"""
        
        return self._send_email(to_email, subject, html_body, text_body)
    
    def send_password_reset_email(
        self,
        to_email: str,
        username: str,
        reset_url: str,
        expire_hours: int = 1
    ) -> bool:
        """
        Send password reset email.
        
        Args:
            to_email: User's email address
            username: User's username
            reset_url: URL for password reset
            expire_hours: Hours until link expires
            
        Returns:
            True if sent successfully
        """
        subject = "Reset your EmergentFolds password"
        
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 8px 8px 0 0; text-align: center;">
        <h1 style="color: white; margin: 0; font-size: 28px;">EmergentFolds</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0;">Protein Structure Prediction</p>
    </div>
    
    <div style="background: #ffffff; padding: 30px; border: 1px solid #e0e0e0; border-top: none; border-radius: 0 0 8px 8px;">
        <h2 style="color: #333; margin-top: 0;">Reset Your Password</h2>
        
        <p>Hi <strong>{username}</strong>,</p>
        
        <p>We received a request to reset your password. Click the button below to choose a new password:</p>
        
        <div style="text-align: center; margin: 30px 0;">
            <a href="{reset_url}" 
               style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                      color: white; 
                      padding: 14px 32px; 
                      text-decoration: none; 
                      border-radius: 6px; 
                      font-weight: bold;
                      display: inline-block;">
                Reset Password
            </a>
        </div>
        
        <p style="color: #666; font-size: 14px;">
            This link will expire in <strong>{expire_hours} hour(s)</strong>.
        </p>
        
        <p style="color: #666; font-size: 14px;">
            If you didn't request a password reset, you can safely ignore this email. Your password will remain unchanged.
        </p>
        
        <hr style="border: none; border-top: 1px solid #e0e0e0; margin: 30px 0;">
        
        <p style="color: #999; font-size: 12px; margin-bottom: 0;">
            If the button doesn't work, copy and paste this link into your browser:
        </p>
        <p style="color: #667eea; font-size: 12px; word-break: break-all; margin-top: 5px;">
            {reset_url}
        </p>
    </div>
    
    <div style="text-align: center; padding: 20px; color: #999; font-size: 12px;">
        <p style="margin: 0;">© 2025 EmergentFolds. All rights reserved.</p>
        <p style="margin: 5px 0 0 0;">
            <a href="https://emergentfolds.com" style="color: #667eea; text-decoration: none;">emergentfolds.com</a>
        </p>
    </div>
</body>
</html>
"""
        
        text_body = f"""
EmergentFolds - Reset Your Password

Hi {username},

We received a request to reset your password. Visit the link below to choose a new password:

{reset_url}

This link will expire in {expire_hours} hour(s).

If you didn't request a password reset, you can safely ignore this email. Your password will remain unchanged.

---
© 2025 EmergentFolds. All rights reserved.
https://emergentfolds.com
"""
        
        return self._send_email(to_email, subject, html_body, text_body)
