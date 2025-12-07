"""Add user quota tracking fields

Revision ID: 002
Revises: 001
Create Date: 2025-12-07 00:00:00.000000

This migration adds quota tracking fields to the users table:
- daily_prediction_count: Current day's prediction count
- monthly_prediction_count: Current month's prediction count
- daily_quota_reset_at: When daily quota was last reset
- monthly_quota_reset_at: When monthly quota was last reset
- account_tier: User's subscription tier (free, pro, enterprise)
- daily_prediction_limit: Max predictions per day
- monthly_prediction_limit: Max predictions per month
"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime, timezone


# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add quota tracking columns to users table"""
    
    # Add quota tracking columns
    op.add_column('users', sa.Column(
        'daily_prediction_count', 
        sa.Integer(), 
        nullable=False, 
        server_default='0'
    ))
    op.add_column('users', sa.Column(
        'monthly_prediction_count', 
        sa.Integer(), 
        nullable=False, 
        server_default='0'
    ))
    op.add_column('users', sa.Column(
        'daily_quota_reset_at', 
        sa.DateTime(timezone=True), 
        nullable=True
    ))
    op.add_column('users', sa.Column(
        'monthly_quota_reset_at', 
        sa.DateTime(timezone=True), 
        nullable=True
    ))
    
    # Add tier settings columns
    op.add_column('users', sa.Column(
        'account_tier', 
        sa.String(length=20), 
        nullable=False, 
        server_default='free'
    ))
    op.add_column('users', sa.Column(
        'daily_prediction_limit', 
        sa.Integer(), 
        nullable=False, 
        server_default='20'
    ))
    op.add_column('users', sa.Column(
        'monthly_prediction_limit', 
        sa.Integer(), 
        nullable=False, 
        server_default='100'
    ))


def downgrade() -> None:
    """Remove quota tracking columns from users table"""
    
    op.drop_column('users', 'monthly_prediction_limit')
    op.drop_column('users', 'daily_prediction_limit')
    op.drop_column('users', 'account_tier')
    op.drop_column('users', 'monthly_quota_reset_at')
    op.drop_column('users', 'daily_quota_reset_at')
    op.drop_column('users', 'monthly_prediction_count')
    op.drop_column('users', 'daily_prediction_count')
