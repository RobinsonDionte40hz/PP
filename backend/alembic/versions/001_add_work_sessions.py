"""Add work sessions and shared exports

Revision ID: 001
Revises: 
Create Date: 2025-11-26 00:00:00.000000

This migration adds:
- work_sessions table for organizing predictions
- shared_exports table for public sharing
- session_id foreign key to predictions table
"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime, timezone


# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create work_sessions and shared_exports tables, add session_id to predictions"""
    
    # Check if users table exists, create it if not
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    
    if 'users' not in inspector.get_table_names():
        op.create_table(
            'users',
            sa.Column('key_id', sa.String(length=36), nullable=False),
            sa.Column('username', sa.String(length=150), nullable=False),
            sa.Column('email', sa.String(length=255), nullable=True),
            sa.Column('password_hash', sa.String(length=255), nullable=False),
            sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
            sa.Column('role', sa.String(length=20), nullable=False, default='user'),
            sa.Column('created_at', sa.DateTime(), nullable=False, default=lambda: datetime.now(timezone.utc)),
            sa.Column('updated_at', sa.DateTime(), nullable=False, default=lambda: datetime.now(timezone.utc)),
            sa.Column('last_login', sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint('key_id')
        )
        op.create_index('ix_users_key_id', 'users', ['key_id'])
        op.create_index('ix_users_username', 'users', ['username'], unique=True)
        op.create_index('ix_users_email', 'users', ['email'], unique=True)
        op.create_index('idx_username_active', 'users', ['username', 'is_active'])
        op.create_index('idx_email_active', 'users', ['email', 'is_active'])
    
    # Create work_sessions table
    op.create_table(
        'work_sessions',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=lambda: datetime.now(timezone.utc)),
        sa.Column('updated_at', sa.DateTime(), nullable=False, default=lambda: datetime.now(timezone.utc)),
        sa.Column('last_active_at', sa.DateTime(), nullable=False, default=lambda: datetime.now(timezone.utc)),
        sa.ForeignKeyConstraint(['user_id'], ['users.key_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for work_sessions
    op.create_index('ix_work_sessions_id', 'work_sessions', ['id'])
    op.create_index('ix_work_sessions_user_id', 'work_sessions', ['user_id'])
    op.create_index('idx_user_last_active', 'work_sessions', ['user_id', 'last_active_at'])
    op.create_index('idx_user_created', 'work_sessions', ['user_id', 'created_at'])
    
    # Create shared_exports table
    op.create_table(
        'shared_exports',
        sa.Column('share_id', sa.String(length=36), nullable=False),
        sa.Column('session_id', sa.String(length=36), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, default=lambda: datetime.now(timezone.utc)),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('access_count', sa.Integer(), nullable=False, default=0),
        sa.Column('last_accessed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['work_sessions.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('share_id')
    )
    
    # Create indexes for shared_exports
    op.create_index('ix_shared_exports_share_id', 'shared_exports', ['share_id'])
    op.create_index('ix_shared_exports_session_id', 'shared_exports', ['session_id'])
    op.create_index('idx_session_expires', 'shared_exports', ['session_id', 'expires_at'])
    op.create_index('idx_expires', 'shared_exports', ['expires_at'])
    
    # Add session_id column to predictions table if it exists
    # Using batch mode for SQLite compatibility
    # Check if predictions table exists first
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    
    if 'predictions' in inspector.get_table_names():
        with op.batch_alter_table('predictions', schema=None) as batch_op:
            batch_op.add_column(sa.Column('session_id', sa.String(length=36), nullable=True))
            batch_op.create_foreign_key(
                'fk_predictions_session_id',
                'work_sessions',
                ['session_id'],
                ['id'],
                ondelete='CASCADE'
            )
            batch_op.create_index('ix_predictions_session_id', ['session_id'])
            batch_op.create_index('idx_session_created', ['session_id', 'created_at'])
            batch_op.create_index('idx_session_status', ['session_id', 'status'])
    else:
        # If predictions table doesn't exist, create it with session_id from the start
        op.create_table(
            'predictions',
            sa.Column('id', sa.String(), nullable=False),
            sa.Column('session_id', sa.String(length=36), nullable=True),
            sa.Column('sequence', sa.Text(), nullable=False),
            sa.Column('status', sa.String(), nullable=False),
            sa.Column('configuration', sa.JSON(), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=False, default=lambda: datetime.now(timezone.utc)),
            sa.Column('updated_at', sa.DateTime(), nullable=False, default=lambda: datetime.now(timezone.utc)),
            sa.Column('started_at', sa.DateTime(), nullable=True),
            sa.Column('completed_at', sa.DateTime(), nullable=True),
            sa.Column('error_message', sa.Text(), nullable=True),
            sa.Column('task_id', sa.String(), nullable=True),
            sa.Column('checkpoint_path', sa.String(), nullable=True),
            sa.Column('result_path', sa.String(), nullable=True),
            sa.Column('current_iteration', sa.Integer(), nullable=False, default=0),
            sa.Column('total_iterations', sa.Integer(), nullable=False, default=0),
            sa.Column('progress_percentage', sa.Float(), nullable=False, default=0.0),
            sa.Column('metrics', sa.JSON(), nullable=False, default=dict),
            sa.ForeignKeyConstraint(['session_id'], ['work_sessions.id'], ondelete='CASCADE'),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index('ix_predictions_id', 'predictions', ['id'])
        op.create_index('ix_predictions_session_id', 'predictions', ['session_id'])
        op.create_index('idx_session_created', 'predictions', ['session_id', 'created_at'])
        op.create_index('idx_session_status', 'predictions', ['session_id', 'status'])


def downgrade() -> None:
    """Remove work_sessions and shared_exports tables, remove session_id from predictions"""
    
    # Check if predictions table exists and if we created it or just modified it
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    
    if 'predictions' in inspector.get_table_names():
        # Check if this migration created the table or just modified it
        # If only the session_id related indexes exist, we created it; otherwise we modified it
        indexes = [idx['name'] for idx in inspector.get_indexes('predictions')]
        
        # If the table has our session indexes, handle accordingly
        if 'idx_session_status' in indexes:
            # Try to drop as batch alter first (for modified tables)
            try:
                with op.batch_alter_table('predictions', schema=None) as batch_op:
                    if 'idx_session_status' in indexes:
                        batch_op.drop_index('idx_session_status')
                    if 'idx_session_created' in indexes:
                        batch_op.drop_index('idx_session_created')
                    if 'ix_predictions_session_id' in indexes:
                        batch_op.drop_index('ix_predictions_session_id')
                    batch_op.drop_constraint('fk_predictions_session_id', type_='foreignkey')
                    batch_op.drop_column('session_id')
            except:
                # If batch alter fails, table was created by this migration, so drop it
                op.drop_table('predictions')
    
    # Drop shared_exports indexes
    op.drop_index('idx_expires', 'shared_exports')
    op.drop_index('idx_session_expires', 'shared_exports')
    op.drop_index('ix_shared_exports_session_id', 'shared_exports')
    op.drop_index('ix_shared_exports_share_id', 'shared_exports')
    
    # Drop shared_exports table
    op.drop_table('shared_exports')
    
    # Drop work_sessions indexes
    op.drop_index('idx_user_created', 'work_sessions')
    op.drop_index('idx_user_last_active', 'work_sessions')
    op.drop_index('ix_work_sessions_user_id', 'work_sessions')
    op.drop_index('ix_work_sessions_id', 'work_sessions')
    
    # Drop work_sessions table
    op.drop_table('work_sessions')
