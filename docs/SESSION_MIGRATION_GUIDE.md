# Work Session Migration Guide

This guide explains how to migrate existing predictions to the new session-based storage system.

## Table of Contents

- [Overview](#overview)
- [What Changed](#what-changed)
- [Migration Strategy](#migration-strategy)
- [Automated Migration Script](#automated-migration-script)
- [Manual Migration](#manual-migration)
- [Backward Compatibility](#backward-compatibility)
- [Validation](#validation)
- [Troubleshooting](#troubleshooting)

---

## Overview

The session-based storage system organizes predictions into logical work sessions with:

- **Isolated file storage** per user and session
- **Improved organization** for related predictions
- **Easy sharing** via time-limited links
- **Automatic cleanup** of expired sessions

**Migration Status**: ✅ Backward compatible - existing predictions continue to work without migration.

---

## What Changed

### Database Changes

**New Tables**:
- `work_sessions`: Stores session metadata (id, user_id, name, timestamps)
- `shared_exports`: Stores share links (share_id, session_id, expires_at)

**Modified Tables**:
- `predictions`: Added `session_id` foreign key (nullable for backward compatibility)

### File Storage Changes

**Old Structure** (still supported):
```
results/
├── {prediction_id}/
│   ├── results.json
│   ├── trajectory.json
│   └── structure.pdb
```

**New Structure**:
```
user_data/
└── {user_id}/
    └── sessions/
        └── {session_id}/
            └── {prediction_id}/
                ├── results.json
                ├── trajectory.json
                ├── structure.pdb
                └── visualization.png
```

### API Changes

**New Endpoints**:
- `GET /api/sessions` - List work sessions
- `POST /api/sessions` - Create work session
- `GET /api/sessions/{id}` - Get session details
- `PUT /api/sessions/{id}` - Update session
- `DELETE /api/sessions/{id}` - Delete session
- `GET /api/sessions/{id}/predictions` - List session predictions
- `POST /api/sessions/{id}/predictions` - Create prediction in session
- `GET /api/sessions/{id}/download` - Download session as ZIP
- `POST /api/sessions/{id}/share` - Create share link
- `GET /api/shared/{share_id}` - Access shared session

**Existing Endpoints**: Remain unchanged and fully functional.

---

## Migration Strategy

You have **three options** for migrating existing predictions:

### Option 1: No Migration (Recommended)

**Best for**: Production systems with minimal disruption.

**Approach**:
- Keep existing predictions as-is (no `session_id`)
- Start using sessions for new predictions
- Old predictions remain accessible via existing endpoints
- No downtime required

**Advantages**:
- ✅ Zero downtime
- ✅ No data migration risks
- ✅ Immediate adoption of new features
- ✅ Backward compatible

**Disadvantages**:
- ❌ Old predictions not organized in sessions
- ❌ Cannot use session features (sharing, ZIP download) for old predictions

### Option 2: Gradual Migration

**Best for**: Users who want to organize important predictions without disrupting workflow.

**Approach**:
1. Create new sessions via API
2. Manually assign important predictions to sessions
3. Use migration script for batch updates
4. Leave unimportant predictions unmigrated

**Advantages**:
- ✅ Flexible migration pace
- ✅ Focus on important data
- ✅ Low risk

**Disadvantages**:
- ❌ Requires manual intervention
- ❌ Mixed state during transition

### Option 3: Full Migration

**Best for**: Test/development environments or clean slate scenarios.

**Approach**:
1. Run database migration (already applied)
2. Run automated migration script
3. Verify all predictions migrated
4. Optionally clean up old file structure

**Advantages**:
- ✅ Clean, consistent state
- ✅ All predictions in sessions
- ✅ Full feature access

**Disadvantages**:
- ❌ Requires downtime
- ❌ More complex rollback

---

## Automated Migration Script

The automated migration script creates a default session for each user and migrates their predictions.

### Script Location

`backend/scripts/migrate_predictions_to_sessions.py`

### Usage

```bash
# Navigate to backend directory
cd backend

# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Run migration script
python scripts/migrate_predictions_to_sessions.py

# Dry run (preview without changes)
python scripts/migrate_predictions_to_sessions.py --dry-run

# Migrate specific user
python scripts/migrate_predictions_to_sessions.py --user-id <user_key_id>
```

### What the Script Does

1. **Identifies unmigrated predictions**: Finds all predictions with `session_id = NULL`
2. **Groups by user**: Organizes predictions by `user_id`
3. **Creates default sessions**: Creates "Migrated Predictions" session for each user
4. **Updates predictions**: Sets `session_id` for each prediction
5. **Moves files** (optional): Relocates files to new structure
6. **Logs progress**: Detailed logging for audit trail

### Script Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dry-run` | Preview changes without applying | False |
| `--user-id` | Migrate specific user only | All users |
| `--move-files` | Move files to new structure | False |
| `--session-name` | Custom session name | "Migrated Predictions" |
| `--batch-size` | Database batch size | 100 |

### Example Output

```
=== Prediction Migration to Sessions ===
Mode: DRY RUN (no changes will be made)

Finding unmigrated predictions...
Found 143 predictions without sessions

Analyzing by user:
  - User user_abc123: 87 predictions
  - User user_def456: 56 predictions

Plan:
  - Create 2 new sessions
  - Update 143 predictions
  - Move 0 files (--move-files not enabled)

DRY RUN COMPLETE - no changes made
Run without --dry-run to apply changes
```

---

## Manual Migration

If you prefer manual control, you can migrate predictions using the API or database.

### Via API

```python
import requests

# 1. Create a session
response = requests.post(
    'http://localhost:8000/api/sessions',
    headers={'Authorization': 'Bearer <token>'},
    json={'name': 'Migrated Studies'}
)
session_id = response.json()['id']

# 2. Update predictions (requires direct DB access)
# Use the migration script or SQL for this step
```

### Via Database (SQL)

```sql
-- 1. Create session for user
INSERT INTO work_sessions (id, user_id, name, created_at, last_active_at)
VALUES (
    'sess_migration_001',
    'user_abc123',
    'Migrated Predictions',
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
);

-- 2. Update predictions to link to session
UPDATE predictions
SET session_id = 'sess_migration_001'
WHERE user_id = 'user_abc123'
  AND session_id IS NULL;

-- 3. Verify migration
SELECT COUNT(*) FROM predictions
WHERE user_id = 'user_abc123' AND session_id = 'sess_migration_001';
```

### Move Files (Python)

```python
import shutil
from pathlib import Path

# Old location
old_path = Path("results/pred_abc123")

# New location
new_path = Path("user_data/user_abc123/sessions/sess_migration_001/pred_abc123")

# Move files
if old_path.exists():
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old_path), str(new_path))
    print(f"Moved {old_path} -> {new_path}")
```

---

## Backward Compatibility

The system maintains full backward compatibility with existing predictions.

### Handling NULL session_id

**Database**: `session_id` is nullable, allowing predictions without sessions.

**API**: Existing prediction endpoints work identically:
- `GET /api/predictions` - Lists all predictions (with or without sessions)
- `GET /api/predictions/{id}` - Works for any prediction
- File storage automatically handles both old and new structures

**File Resolution**:
```python
def get_prediction_files(prediction_id, user_id, session_id=None):
    if session_id:
        # New session-based path
        path = f"user_data/{user_id}/sessions/{session_id}/{prediction_id}/"
    else:
        # Old path for backward compatibility
        path = f"results/{prediction_id}/"
    
    return path
```

### Frontend Compatibility

The frontend handles both types seamlessly:
- Predictions with sessions show session name
- Predictions without sessions show "No Session"
- All prediction features work regardless of session status

---

## Validation

After migration, validate the results:

### 1. Database Validation

```sql
-- Check all predictions have sessions
SELECT COUNT(*) as unmigrated_count
FROM predictions
WHERE session_id IS NULL;

-- Verify session integrity
SELECT 
    ws.id as session_id,
    ws.name,
    COUNT(p.id) as prediction_count
FROM work_sessions ws
LEFT JOIN predictions p ON p.session_id = ws.id
GROUP BY ws.id, ws.name;

-- Check for orphaned predictions (user mismatch)
SELECT p.id, p.user_id, p.session_id, ws.user_id as session_user_id
FROM predictions p
JOIN work_sessions ws ON p.session_id = ws.id
WHERE p.user_id != ws.user_id;
```

### 2. File System Validation

```bash
# Check old structure (should be empty after full migration)
ls -la results/

# Check new structure
ls -la user_data/*/sessions/*/

# Compare file counts
find results -type f | wc -l
find user_data -type f | wc -l
```

### 3. API Validation

```python
import requests

headers = {'Authorization': 'Bearer <token>'}

# List sessions
sessions = requests.get('http://localhost:8000/api/sessions', headers=headers).json()
print(f"Total sessions: {sessions['pagination']['total']}")

# Check each session
for session in sessions['sessions']:
    session_id = session['id']
    
    # Get session details
    details = requests.get(
        f'http://localhost:8000/api/sessions/{session_id}',
        headers=headers
    ).json()
    
    print(f"Session {session_id}: {details['prediction_count']} predictions")
    
    # List predictions in session
    predictions = requests.get(
        f'http://localhost:8000/api/sessions/{session_id}/predictions',
        headers=headers
    ).json()
    
    print(f"  - Retrieved {len(predictions['predictions'])} prediction records")
```

---

## Troubleshooting

### Migration Script Fails

**Error**: `IntegrityError: FOREIGN KEY constraint failed`

**Cause**: Session doesn't exist for user_id

**Solution**:
```python
# Create session first
from app.services.work_session_service import work_session_service
session = work_session_service.create_session(
    user_id="user_abc123",
    name="Migrated Predictions"
)
```

---

### File Not Found After Migration

**Error**: `FileNotFoundError: [Errno 2] No such file or directory`

**Cause**: Files not moved to new location

**Solution**: Run migration script with `--move-files` option:
```bash
python scripts/migrate_predictions_to_sessions.py --move-files
```

---

### Session Ownership Mismatch

**Error**: `403 Forbidden: Session belongs to another user`

**Cause**: Prediction's `user_id` doesn't match session's `user_id`

**Solution**: Fix ownership in database:
```sql
UPDATE predictions
SET session_id = (
    SELECT id FROM work_sessions
    WHERE user_id = predictions.user_id
    LIMIT 1
)
WHERE id = 'pred_abc123';
```

---

### Old Predictions Not Showing

**Issue**: Legacy predictions (without session_id) not appearing in session views

**Cause**: This is expected behavior - sessions only show predictions with `session_id`

**Solution**: Either:
1. Migrate predictions to a session (recommended)
2. Use `/api/predictions` endpoint to list all predictions (with and without sessions)

---

### Permission Denied on File Move

**Error**: `PermissionError: [Errno 13] Permission denied`

**Cause**: Insufficient file system permissions

**Solution**: Run migration with appropriate permissions:
```bash
# Linux/Mac
sudo python scripts/migrate_predictions_to_sessions.py --move-files

# Windows (run as Administrator)
# Right-click PowerShell -> "Run as Administrator"
python scripts/migrate_predictions_to_sessions.py --move-files
```

---

## Migration Script Implementation

Here's the complete migration script:

```python
"""
Migrate existing predictions to session-based storage

Usage:
    python scripts/migrate_predictions_to_sessions.py
    python scripts/migrate_predictions_to_sessions.py --dry-run
    python scripts/migrate_predictions_to_sessions.py --user-id user_abc123
    python scripts/migrate_predictions_to_sessions.py --move-files
"""
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
import shutil
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from app.database import get_db
from app.models.work_session import WorkSession
from app.models.prediction import Prediction
from app.services.work_session_service import work_session_service
from app.services.file_storage_service import FileStorageService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PredictionMigrator:
    """Migrates predictions to session-based storage"""
    
    def __init__(
        self,
        db: Session,
        dry_run: bool = False,
        move_files: bool = False
    ):
        self.db = db
        self.dry_run = dry_run
        self.move_files = move_files
        self.file_storage = FileStorageService()
        
        self.stats = {
            "unmigrated_predictions": 0,
            "users_affected": 0,
            "sessions_created": 0,
            "predictions_updated": 0,
            "files_moved": 0,
            "errors": []
        }
    
    def run(
        self,
        user_id: Optional[str] = None,
        session_name: str = "Migrated Predictions",
        batch_size: int = 100
    ) -> Dict:
        """
        Run migration process
        
        Args:
            user_id: Migrate specific user only (None = all users)
            session_name: Name for created sessions
            batch_size: Database batch size
            
        Returns:
            Dictionary with migration statistics
        """
        logger.info("=== Prediction Migration to Sessions ===")
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        logger.info(f"Move files: {self.move_files}")
        logger.info("")
        
        # Step 1: Find unmigrated predictions
        logger.info("Finding unmigrated predictions...")
        unmigrated = self._find_unmigrated_predictions(user_id)
        self.stats["unmigrated_predictions"] = len(unmigrated)
        logger.info(f"Found {len(unmigrated)} predictions without sessions")
        
        if not unmigrated:
            logger.info("No predictions to migrate")
            return self.stats
        
        # Step 2: Group by user
        logger.info("")
        logger.info("Analyzing by user:")
        by_user = self._group_by_user(unmigrated)
        self.stats["users_affected"] = len(by_user)
        
        for uid, preds in by_user.items():
            logger.info(f"  - User {uid}: {len(preds)} predictions")
        
        # Step 3: Create sessions and migrate
        logger.info("")
        logger.info("Plan:")
        logger.info(f"  - Create {len(by_user)} new sessions")
        logger.info(f"  - Update {len(unmigrated)} predictions")
        if self.move_files:
            logger.info(f"  - Move files to new structure")
        else:
            logger.info(f"  - Keep files in current location")
        
        if self.dry_run:
            logger.info("")
            logger.info("DRY RUN COMPLETE - no changes made")
            logger.info("Run without --dry-run to apply changes")
            return self.stats
        
        logger.info("")
        logger.info("Starting migration...")
        
        for uid, predictions in by_user.items():
            self._migrate_user_predictions(
                user_id=uid,
                predictions=predictions,
                session_name=session_name
            )
        
        logger.info("")
        logger.info("=== Migration Complete ===")
        logger.info(f"Sessions created: {self.stats['sessions_created']}")
        logger.info(f"Predictions updated: {self.stats['predictions_updated']}")
        logger.info(f"Files moved: {self.stats['files_moved']}")
        if self.stats["errors"]:
            logger.warning(f"Errors encountered: {len(self.stats['errors'])}")
            for error in self.stats["errors"]:
                logger.warning(f"  - {error}")
        
        return self.stats
    
    def _find_unmigrated_predictions(
        self,
        user_id: Optional[str] = None
    ) -> List[Prediction]:
        """Find predictions without session_id"""
        query = self.db.query(Prediction).filter(
            Prediction.session_id.is_(None)
        )
        
        if user_id:
            query = query.filter(Prediction.user_id == user_id)
        
        return query.all()
    
    def _group_by_user(
        self,
        predictions: List[Prediction]
    ) -> Dict[str, List[Prediction]]:
        """Group predictions by user_id"""
        by_user = {}
        for pred in predictions:
            if pred.user_id not in by_user:
                by_user[pred.user_id] = []
            by_user[pred.user_id].append(pred)
        return by_user
    
    def _migrate_user_predictions(
        self,
        user_id: str,
        predictions: List[Prediction],
        session_name: str
    ):
        """Migrate all predictions for a user"""
        try:
            # Create session
            logger.info(f"Creating session for user {user_id}...")
            session = work_session_service.create_session(
                user_id=user_id,
                name=session_name,
                db=self.db
            )
            self.stats["sessions_created"] += 1
            logger.info(f"  Created session {session.id}")
            
            # Update predictions
            for pred in predictions:
                try:
                    pred.session_id = session.id
                    self.db.commit()
                    self.stats["predictions_updated"] += 1
                    
                    logger.info(f"  Updated prediction {pred.id}")
                    
                    # Move files if requested
                    if self.move_files:
                        moved = self._move_prediction_files(
                            prediction_id=pred.id,
                            user_id=user_id,
                            session_id=session.id
                        )
                        if moved:
                            self.stats["files_moved"] += 1
                
                except Exception as e:
                    error_msg = f"Failed to migrate prediction {pred.id}: {str(e)}"
                    logger.error(error_msg)
                    self.stats["errors"].append(error_msg)
                    self.db.rollback()
            
        except Exception as e:
            error_msg = f"Failed to create session for user {user_id}: {str(e)}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            self.db.rollback()
    
    def _move_prediction_files(
        self,
        prediction_id: str,
        user_id: str,
        session_id: str
    ) -> bool:
        """Move files from old to new location"""
        try:
            # Old location
            old_path = Path(f"results/{prediction_id}")
            
            if not old_path.exists():
                logger.debug(f"    No files to move for {prediction_id}")
                return False
            
            # New location
            new_path = self.file_storage.get_prediction_directory(
                user_id=user_id,
                session_id=session_id,
                prediction_id=prediction_id
            )
            
            # Move files
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(new_path))
            
            logger.info(f"    Moved files {old_path} -> {new_path}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to move files for {prediction_id}: {str(e)}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrate existing predictions to session-based storage"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying"
    )
    parser.add_argument(
        "--user-id",
        type=str,
        help="Migrate specific user only"
    )
    parser.add_argument(
        "--move-files",
        action="store_true",
        help="Move files to new structure"
    )
    parser.add_argument(
        "--session-name",
        type=str,
        default="Migrated Predictions",
        help="Custom session name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Database batch size"
    )
    
    args = parser.parse_args()
    
    # Get database session
    db = next(get_db())
    
    try:
        # Run migration
        migrator = PredictionMigrator(
            db=db,
            dry_run=args.dry_run,
            move_files=args.move_files
        )
        
        stats = migrator.run(
            user_id=args.user_id,
            session_name=args.session_name,
            batch_size=args.batch_size
        )
        
        # Return success exit code
        return 0 if not stats["errors"] else 1
        
    finally:
        db.close()


if __name__ == "__main__":
    exit(main())
```

Save this script as `backend/scripts/migrate_predictions_to_sessions.py`.

---

## Best Practices

### 1. Always Test First

Run with `--dry-run` to preview changes:
```bash
python scripts/migrate_predictions_to_sessions.py --dry-run
```

### 2. Backup Database

Before migration, backup your database:
```bash
# SQLite
cp pp_dev.db pp_dev.db.backup

# PostgreSQL
pg_dump dbname > backup.sql
```

### 3. Backup Files

Before moving files, create backup:
```bash
cp -r results results_backup
cp -r user_data user_data_backup
```

### 4. Migrate During Low Traffic

Schedule migration during maintenance window to minimize disruption.

### 5. Monitor Logs

Watch migration logs for errors:
```bash
python scripts/migrate_predictions_to_sessions.py 2>&1 | tee migration.log
```

### 6. Validate After Migration

Run validation queries to ensure data integrity (see [Validation](#validation) section).

---

## Rollback

If migration fails, you can rollback:

### 1. Restore Database

```bash
# SQLite
cp pp_dev.db.backup pp_dev.db

# PostgreSQL
psql dbname < backup.sql
```

### 2. Restore Files

```bash
rm -rf user_data
mv user_data_backup user_data

rm -rf results
mv results_backup results
```

### 3. Verify Rollback

```sql
-- Check session_id values
SELECT COUNT(*) FROM predictions WHERE session_id IS NOT NULL;

-- Should be 0 if rollback successful
```

---

## Support

For questions or issues:

1. **Check logs**: Review migration log output
2. **Validate data**: Run validation queries
3. **Test in development**: Migrate dev environment first
4. **Create issue**: Report problems with full logs

---

**Document Version**: 1.0.0  
**Last Updated**: November 26, 2025  
**Compatibility**: Backend v1.0.0+
