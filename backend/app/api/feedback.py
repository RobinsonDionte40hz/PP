"""
Feedback API routes for EmergentFolds.
Handles user feedback submission and email delivery.
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, EmailStr
from typing import Optional
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

from app.config import settings
from app.security import get_current_user
from typing import Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter()


class FeedbackRequest(BaseModel):
    """Feedback submission request model."""
    category: str = Field(..., description="Feedback category", pattern="^(bug|feature|improvement|other)$")
    subject: str = Field(..., min_length=5, max_length=200, description="Feedback subject")
    message: str = Field(..., min_length=10, max_length=5000, description="Feedback message")
    email: Optional[EmailStr] = Field(None, description="Optional email for follow-up")
    include_system_info: bool = Field(False, description="Include system info with feedback")


class FeedbackResponse(BaseModel):
    """Feedback submission response model."""
    success: bool
    message: str
    feedback_id: str


def send_feedback_email(
    category: str,
    subject: str,
    message: str,
    user_email: Optional[str] = None,
    username: Optional[str] = None,
    system_info: Optional[dict] = None
):
    """Send feedback email to the configured recipient."""
    
    # Get email configuration from settings
    smtp_host: Optional[str] = getattr(settings, 'SMTP_HOST', None)
    smtp_port: int = getattr(settings, 'SMTP_PORT', 587)
    smtp_user: Optional[str] = getattr(settings, 'SMTP_USER', None)
    smtp_password: Optional[str] = getattr(settings, 'SMTP_PASSWORD', None)
    feedback_recipient: Optional[str] = getattr(settings, 'FEEDBACK_EMAIL', None)
    
    if not all([smtp_host, smtp_user, smtp_password, feedback_recipient]):
        logger.warning("Email configuration incomplete. Feedback logged but not emailed.")
        # Log the feedback even if email is not configured
        logger.info(f"FEEDBACK [{category.upper()}]: {subject}")
        logger.info(f"From: {username or 'Anonymous'} ({user_email or 'No email'})")
        logger.info(f"Message: {message[:500]}...")
        return False
    
    # Type narrowing - we know these are not None after the check above
    assert smtp_host is not None
    assert smtp_user is not None
    assert smtp_password is not None
    assert feedback_recipient is not None
    
    try:
        # Create email
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[EmergentFolds Feedback] [{category.upper()}] {subject}"
        msg['From'] = smtp_user
        msg['To'] = feedback_recipient
        
        # Build email body
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Plain text version
        text_body = f"""
EmergentFolds User Feedback
===========================

Category: {category.upper()}
Subject: {subject}
Date: {timestamp}

From: {username or 'Anonymous User'}
Reply Email: {user_email or 'Not provided'}

Message:
--------
{message}

"""
        
        if system_info:
            text_body += f"""
System Information:
------------------
Browser: {system_info.get('browser', 'Unknown')}
Platform: {system_info.get('platform', 'Unknown')}
Screen: {system_info.get('screen', 'Unknown')}
"""
        
        # HTML version
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #293B5F 0%, #47597E 100%); color: white; padding: 20px; border-radius: 8px 8px 0 0; }}
        .content {{ background: #f9f9f9; padding: 20px; border: 1px solid #ddd; border-top: none; }}
        .category {{ display: inline-block; background: #B2AB8C; color: #293B5F; padding: 4px 12px; border-radius: 4px; font-weight: bold; text-transform: uppercase; font-size: 12px; }}
        .field {{ margin-bottom: 15px; }}
        .label {{ font-weight: bold; color: #47597E; }}
        .message-box {{ background: white; padding: 15px; border-left: 4px solid #47597E; margin: 15px 0; }}
        .system-info {{ background: #e9e9e9; padding: 10px; border-radius: 4px; font-size: 12px; color: #666; }}
        .footer {{ text-align: center; padding: 15px; color: #888; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2 style="margin: 0;">🧬 EmergentFolds Feedback</h2>
            <p style="margin: 5px 0 0 0; opacity: 0.9;">{timestamp}</p>
        </div>
        <div class="content">
            <div class="field">
                <span class="category">{category}</span>
            </div>
            <div class="field">
                <div class="label">Subject</div>
                <div>{subject}</div>
            </div>
            <div class="field">
                <div class="label">From</div>
                <div>{username or 'Anonymous User'} {f'({user_email})' if user_email else ''}</div>
            </div>
            <div class="field">
                <div class="label">Message</div>
                <div class="message-box">{message.replace(chr(10), '<br>')}</div>
            </div>
            {"<div class='system-info'><strong>System:</strong> " + f"{system_info.get('browser', 'Unknown')} on {system_info.get('platform', 'Unknown')}</div>" if system_info else ""}
        </div>
        <div class="footer">
            EmergentFolds - Quantum-Enhanced Protein Structure Prediction
        </div>
    </div>
</body>
</html>
"""
        
        part1 = MIMEText(text_body, 'plain')
        part2 = MIMEText(html_body, 'html')
        msg.attach(part1)
        msg.attach(part2)
        
        # Send email
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        logger.info(f"Feedback email sent successfully: [{category}] {subject}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send feedback email: {e}")
        # Still log the feedback
        logger.info(f"FEEDBACK [{category.upper()}]: {subject}")
        logger.info(f"From: {username or 'Anonymous'} ({user_email or 'No email'})")
        logger.info(f"Message: {message[:500]}...")
        return False


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """
    Submit user feedback.
    
    Accepts feedback from both authenticated and anonymous users.
    Sends the feedback to the configured email address.
    """
    import uuid
    
    # Generate feedback ID
    feedback_id = f"FB-{datetime.utcnow().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
    
    # Get user info if authenticated
    username = current_user.get("sub") if current_user else None
    user_email = feedback.email or (current_user.get("email") if current_user else None)
    
    # Prepare system info if requested
    system_info = None
    if feedback.include_system_info:
        system_info = {
            "browser": "User Agent from request",
            "platform": "Extracted from headers",
            "screen": "Not available server-side"
        }
    
    # Send email in background to not block the response
    background_tasks.add_task(
        send_feedback_email,
        category=feedback.category,
        subject=feedback.subject,
        message=feedback.message,
        user_email=user_email,
        username=username,
        system_info=system_info
    )
    
    logger.info(f"Feedback received: {feedback_id} from {username or 'anonymous'}")
    
    return FeedbackResponse(
        success=True,
        message="Thank you for your feedback! We appreciate your input.",
        feedback_id=feedback_id
    )


class ContactRequest(BaseModel):
    """Contact form submission request model."""
    name: str = Field(..., min_length=2, max_length=100, description="Contact name")
    email: EmailStr = Field(..., description="Contact email address")
    message: str = Field(..., min_length=10, max_length=5000, description="Contact message")


class ContactResponse(BaseModel):
    """Contact form submission response model."""
    success: bool
    message: str


def send_contact_email(
    name: str,
    email: str,
    message: str
):
    """Send contact form email to the configured recipient."""
    
    # Get email configuration from settings
    smtp_host: Optional[str] = getattr(settings, 'SMTP_HOST', None)
    smtp_port: int = getattr(settings, 'SMTP_PORT', 587)
    smtp_user: Optional[str] = getattr(settings, 'SMTP_USER', None)
    smtp_password: Optional[str] = getattr(settings, 'SMTP_PASSWORD', None)
    feedback_recipient: Optional[str] = getattr(settings, 'FEEDBACK_EMAIL', None)
    
    if not all([smtp_host, smtp_user, smtp_password, feedback_recipient]):
        logger.warning("Email configuration incomplete. Contact logged but not emailed.")
        logger.info(f"CONTACT from {name} ({email})")
        logger.info(f"Message: {message[:500]}...")
        return False
    
    # Type narrowing
    assert smtp_host is not None
    assert smtp_user is not None
    assert smtp_password is not None
    assert feedback_recipient is not None
    
    try:
        # Create email
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[EmergentFolds Contact] Message from {name}"
        msg['From'] = smtp_user
        msg['To'] = feedback_recipient
        msg['Reply-To'] = email
        
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Plain text version
        text_content = f"""
New Contact Form Submission
===========================

From: {name}
Email: {email}
Time: {timestamp}

Message:
--------
{message}
"""
        
        # HTML version
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #293B5F 0%, #47597E 100%); color: white; padding: 20px; border-radius: 8px 8px 0 0; }}
        .content {{ background: #f9f9f9; padding: 20px; border: 1px solid #ddd; border-top: none; border-radius: 0 0 8px 8px; }}
        .info {{ background: white; padding: 15px; border-radius: 4px; margin-bottom: 15px; }}
        .info-row {{ margin: 8px 0; }}
        .label {{ font-weight: bold; color: #666; }}
        .message {{ background: white; padding: 15px; border-radius: 4px; white-space: pre-wrap; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2 style="margin: 0;">📧 New Contact Form Submission</h2>
        </div>
        <div class="content">
            <div class="info">
                <div class="info-row"><span class="label">From:</span> {name}</div>
                <div class="info-row"><span class="label">Email:</span> <a href="mailto:{email}">{email}</a></div>
                <div class="info-row"><span class="label">Time:</span> {timestamp}</div>
            </div>
            <h3>Message:</h3>
            <div class="message">{message}</div>
        </div>
    </div>
</body>
</html>
"""
        
        text_part = MIMEText(text_content, 'plain')
        html_part = MIMEText(html_content, 'html')
        msg.attach(text_part)
        msg.attach(html_part)
        
        # Send email
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        logger.info(f"Contact email sent successfully from {name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send contact email: {str(e)}")
        return False


@router.post("/contact", response_model=ContactResponse)
async def submit_contact(
    contact: ContactRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit contact form message.
    
    Public endpoint - no authentication required.
    Sends the message to the configured email address.
    """
    # Send email in background
    background_tasks.add_task(
        send_contact_email,
        name=contact.name,
        email=contact.email,
        message=contact.message
    )
    
    logger.info(f"Contact form received from {contact.name} ({contact.email})")
    
    return ContactResponse(
        success=True,
        message="Thank you for reaching out! We'll get back to you soon."
    )
