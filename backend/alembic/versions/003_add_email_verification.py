"""Add email verification fields

Revision ID: 003
Revises: 002
Create Date: 2025-12-07 12:00:00.000000

This migration adds email verification fields to the users table:
- email_verified: Whether the user's email has been verified
- email_verification_token: Token sent in verification email
- email_verification_sent_at: When verification email was last sent
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add email verification columns to users table"""
    
    # Add email verification columns
    op.add_column('users', sa.Column(
        'email_verified', 
        sa.Boolean(), 
        nullable=False, 
        server_default='0'  # False by default
    ))
    op.add_column('users', sa.Column(
        'email_verification_token', 
        sa.String(length=64), 
        nullable=True
    ))
    op.add_column('users', sa.Column(
        'email_verification_sent_at', 
        sa.DateTime(timezone=True), 
        nullable=True
    ))
    
    # Create index on verification token for faster lookups
    op.create_index(
        'idx_email_verification_token',
        'users',
        ['email_verification_token'],
        unique=False
    )


def downgrade() -> None:
    """Remove email verification columns from users table"""
    
    # Drop index first
    op.drop_index('idx_email_verification_token', table_name='users')
    
    # Remove columns
    op.drop_column('users', 'email_verification_sent_at')
    op.drop_column('users', 'email_verification_token')
    op.drop_column('users', 'email_verified')
