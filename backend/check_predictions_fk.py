"""
Check if predictions table has the foreign key constraint
"""
from sqlalchemy import create_engine, text, inspect

engine = create_engine('sqlite:///./pp_dev.db')
inspector = inspect(engine)

# List all tables
tables = inspector.get_table_names()
print("Tables in database:")
for table in tables:
    print(f"  - {table}")
print()

# Check if predictions table exists
if 'predictions' not in tables:
    print("predictions table does NOT exist!")
    print("The migration may not have been applied correctly.")
    exit(1)

# Get table creation SQL
with engine.connect() as conn:
    result = conn.execute(text("SELECT sql FROM sqlite_master WHERE type='table' AND name='predictions'"))
    row = result.fetchone()
    if row:
        table_sql = row[0]
        print("Predictions table SQL:")
        print(table_sql)
        print()

# Get foreign keys using inspector
fks = inspector.get_foreign_keys('predictions')
print("Foreign keys on predictions table:")
if fks:
    for fk in fks:
        print(f"  ✓ Column: {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}")
        print(f"    On Delete: {fk.get('ondelete', 'NO ACTION')}")
else:
    print("  ✗ No foreign keys found!")
print()

# Check if session_id column exists
columns = inspector.get_columns('predictions')
session_id_col = [col for col in columns if col['name'] == 'session_id']
if session_id_col:
    print("✓ session_id column exists:")
    print(f"  Type: {session_id_col[0]['type']}")
    print(f"  Nullable: {session_id_col[0]['nullable']}")
else:
    print("✗ session_id column NOT FOUND!")
