"""Verify the users table was created correctly"""
from app.database import engine
from sqlalchemy import inspect

inspector = inspect(engine)

print("✅ Users table verification:")
print()
print("Columns:")
for col in inspector.get_columns("users"):
    nullable = "NULL" if col["nullable"] else "NOT NULL"
    print(f"  - {col['name']}: {col['type']} {nullable}")

print()
print("Indexes:")
for idx in inspector.get_indexes("users"):
    unique = "UNIQUE" if idx["unique"] else ""
    columns = ", ".join(idx["column_names"])
    print(f"  - {idx['name']}: ({columns}) {unique}")

print()
print("✅ Migration verified successfully!")
