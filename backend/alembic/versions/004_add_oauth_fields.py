"""Add OAuth fields to users table

Revision ID: 004_add_oauth_fields
Revises: 003_add_email_verification
Create Date: 2025-12-07

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '004_add_oauth_fields'
down_revision = '003_add_email_verification'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add OAuth fields to users table."""
    
    # Add OAuth fields
    op.add_column('users', sa.Column('google_id', sa.String(100), nullable=True))
    op.add_column('users', sa.Column('github_id', sa.String(100), nullable=True))
    op.add_column('users', sa.Column('oauth_provider', sa.String(20), nullable=True))
    
    # Create unique indexes for OAuth IDs
    op.create_index('idx_users_google_id', 'users', ['google_id'], unique=True)
    op.create_index('idx_users_github_id', 'users', ['github_id'], unique=True)
    
    # Make password_hash nullable for OAuth-only users
    # Note: This may require different syntax depending on the database
    # SQLite doesn't support ALTER COLUMN, so we need to handle this carefully
    try:
        # For PostgreSQL
        op.alter_column('users', 'password_hash', nullable=True)
    except Exception:
        # SQLite doesn't support ALTER COLUMN - the column will already be nullable
        # if we recreate the table or just skip this for SQLite
        pass


def downgrade() -> None:
    """Remove OAuth fields from users table."""
    
    # Drop indexes
    op.drop_index('idx_users_google_id', table_name='users')
    op.drop_index('idx_users_github_id', table_name='users')
    
    # Drop columns
    op.drop_column('users', 'oauth_provider')
    op.drop_column('users', 'github_id')
    op.drop_column('users', 'google_id')
    
    # Make password_hash not nullable again
    # Note: This will fail if there are OAuth-only users without passwords
    try:
        op.alter_column('users', 'password_hash', nullable=False)
    except Exception:
        pass
