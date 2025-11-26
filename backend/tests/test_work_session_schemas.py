"""
Tests for work session schema validation
"""
import pytest
from datetime import datetime, timedelta, timezone
from pydantic import ValidationError

from app.schemas.work_session import (
    WorkSessionCreateSchema,
    WorkSessionUpdateSchema,
    WorkSessionResponseSchema,
    WorkSessionListResponseSchema,
    ShareLinkCreateSchema,
    ShareLinkResponseSchema,
    SharedSessionResponseSchema,
)


class TestWorkSessionCreateSchema:
    """Tests for WorkSessionCreateSchema validation"""
    
    def test_valid_session_name(self):
        """Test valid session names"""
        # Basic name
        schema = WorkSessionCreateSchema(name="My Project")
        assert schema.name == "My Project"
        
        # Name with special characters
        schema = WorkSessionCreateSchema(name="Project-2025 (v1.0)")
        assert schema.name == "Project-2025 (v1.0)"
        
        # Long name (under 255 chars)
        long_name = "A" * 255
        schema = WorkSessionCreateSchema(name=long_name)
        assert schema.name == long_name
    
    def test_whitespace_trimming(self):
        """Test that whitespace is trimmed from session names"""
        schema = WorkSessionCreateSchema(name="  Trimmed Name  ")
        assert schema.name == "Trimmed Name"
        
        schema = WorkSessionCreateSchema(name="\t\nSpaced\t\n")
        assert schema.name == "Spaced"
    
    def test_empty_name_rejected(self):
        """Test that empty names are rejected"""
        with pytest.raises(ValidationError) as exc_info:
            WorkSessionCreateSchema(name="")
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "name" in str(errors[0])
        # Pydantic validates empty string as "string_too_short" before custom validator runs
        assert "string_too_short" in str(errors[0]) or "empty" in str(errors[0]).lower()
    
    def test_whitespace_only_rejected(self):
        """Test that whitespace-only names are rejected"""
        with pytest.raises(ValidationError) as exc_info:
            WorkSessionCreateSchema(name="   ")
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "empty" in str(errors[0]).lower()
        
        with pytest.raises(ValidationError) as exc_info:
            WorkSessionCreateSchema(name="\t\n  \t")
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
    
    def test_name_too_long_rejected(self):
        """Test that names exceeding 255 characters are rejected"""
        # Exactly 256 chars
        too_long = "A" * 256
        with pytest.raises(ValidationError) as exc_info:
            WorkSessionCreateSchema(name=too_long)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "255" in str(errors[0])
    
    def test_missing_name_rejected(self):
        """Test that missing name is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            WorkSessionCreateSchema()
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "name" in str(errors[0])


class TestWorkSessionUpdateSchema:
    """Tests for WorkSessionUpdateSchema validation"""
    
    def test_valid_name_update(self):
        """Test valid name updates"""
        schema = WorkSessionUpdateSchema(name="Updated Name")
        assert schema.name == "Updated Name"
    
    def test_empty_name_rejected(self):
        """Test that empty names are rejected"""
        with pytest.raises(ValidationError) as exc_info:
            WorkSessionUpdateSchema(name="")
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        # Pydantic validates empty string as "string_too_short" before custom validator runs
        assert "string_too_short" in str(errors[0]) or "empty" in str(errors[0]).lower()
    
    def test_name_too_long_rejected(self):
        """Test that names exceeding 255 characters are rejected"""
        too_long = "A" * 256
        with pytest.raises(ValidationError) as exc_info:
            WorkSessionUpdateSchema(name=too_long)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1


class TestWorkSessionResponseSchema:
    """Tests for WorkSessionResponseSchema"""
    
    def test_valid_response_schema(self):
        """Test valid response schema creation"""
        now = datetime.now(timezone.utc)
        schema = WorkSessionResponseSchema(
            id="550e8400-e29b-41d4-a716-446655440000",
            user_id="660e8400-e29b-41d4-a716-446655440111",
            name="Test Session",
            created_at=now,
            updated_at=now,
            last_active_at=now,
            prediction_count=5,
            total_size=1024000
        )
        
        assert schema.id == "550e8400-e29b-41d4-a716-446655440000"
        assert schema.user_id == "660e8400-e29b-41d4-a716-446655440111"
        assert schema.name == "Test Session"
        assert schema.prediction_count == 5
        assert schema.total_size == 1024000
    
    def test_optional_fields_default(self):
        """Test that optional fields have default values"""
        now = datetime.now(timezone.utc)
        schema = WorkSessionResponseSchema(
            id="550e8400-e29b-41d4-a716-446655440000",
            user_id="660e8400-e29b-41d4-a716-446655440111",
            name="Test Session",
            created_at=now,
            updated_at=now,
            last_active_at=now
        )
        
        assert schema.prediction_count == 0
        assert schema.total_size == 0
    
    def test_serialization(self):
        """Test JSON serialization"""
        now = datetime.now(timezone.utc)
        schema = WorkSessionResponseSchema(
            id="550e8400-e29b-41d4-a716-446655440000",
            user_id="660e8400-e29b-41d4-a716-446655440111",
            name="Test Session",
            created_at=now,
            updated_at=now,
            last_active_at=now,
            prediction_count=5,
            total_size=1024000
        )
        
        json_data = schema.model_dump()
        assert "id" in json_data
        assert "user_id" in json_data
        assert "name" in json_data
        assert "created_at" in json_data


class TestWorkSessionListResponseSchema:
    """Tests for WorkSessionListResponseSchema"""
    
    def test_valid_list_response(self):
        """Test valid list response"""
        now = datetime.now(timezone.utc)
        session = WorkSessionResponseSchema(
            id="550e8400-e29b-41d4-a716-446655440000",
            user_id="660e8400-e29b-41d4-a716-446655440111",
            name="Test Session",
            created_at=now,
            updated_at=now,
            last_active_at=now,
            prediction_count=5,
            total_size=1024000
        )
        
        schema = WorkSessionListResponseSchema(
            sessions=[session],
            total=1,
            page=1,
            page_size=20
        )
        
        assert len(schema.sessions) == 1
        assert schema.total == 1
        assert schema.page == 1
        assert schema.page_size == 20
    
    def test_empty_list(self):
        """Test empty session list"""
        schema = WorkSessionListResponseSchema(
            sessions=[],
            total=0,
            page=1,
            page_size=20
        )
        
        assert len(schema.sessions) == 0
        assert schema.total == 0
    
    def test_multiple_sessions(self):
        """Test list with multiple sessions"""
        now = datetime.now(timezone.utc)
        sessions = []
        for i in range(3):
            session = WorkSessionResponseSchema(
                id=f"550e8400-e29b-41d4-a716-44665544000{i}",
                user_id="660e8400-e29b-41d4-a716-446655440111",
                name=f"Session {i}",
                created_at=now,
                updated_at=now,
                last_active_at=now,
                prediction_count=i,
                total_size=1024000 * i
            )
            sessions.append(session)
        
        schema = WorkSessionListResponseSchema(
            sessions=sessions,
            total=3,
            page=1,
            page_size=20
        )
        
        assert len(schema.sessions) == 3
        assert schema.total == 3


class TestShareLinkCreateSchema:
    """Tests for ShareLinkCreateSchema validation"""
    
    def test_valid_expiration_hours(self):
        """Test valid expiration hours"""
        # Minimum (1 hour)
        schema = ShareLinkCreateSchema(expiration_hours=1)
        assert schema.expiration_hours == 1
        
        # Common values
        schema = ShareLinkCreateSchema(expiration_hours=24)
        assert schema.expiration_hours == 24
        
        schema = ShareLinkCreateSchema(expiration_hours=72)
        assert schema.expiration_hours == 72
        
        # Maximum (168 hours = 7 days)
        schema = ShareLinkCreateSchema(expiration_hours=168)
        assert schema.expiration_hours == 168
    
    def test_expiration_too_low_rejected(self):
        """Test that expiration below 1 hour is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            ShareLinkCreateSchema(expiration_hours=0)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "1" in str(errors[0])
        
        with pytest.raises(ValidationError) as exc_info:
            ShareLinkCreateSchema(expiration_hours=-1)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
    
    def test_expiration_too_high_rejected(self):
        """Test that expiration above 168 hours is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            ShareLinkCreateSchema(expiration_hours=169)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "168" in str(errors[0])
        
        with pytest.raises(ValidationError) as exc_info:
            ShareLinkCreateSchema(expiration_hours=1000)
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
    
    def test_missing_expiration_rejected(self):
        """Test that missing expiration is rejected"""
        with pytest.raises(ValidationError) as exc_info:
            ShareLinkCreateSchema()
        
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "expiration_hours" in str(errors[0])


class TestShareLinkResponseSchema:
    """Tests for ShareLinkResponseSchema"""
    
    def test_valid_share_link_response(self):
        """Test valid share link response"""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=24)
        
        schema = ShareLinkResponseSchema(
            share_id="770e8400-e29b-41d4-a716-446655440222",
            session_id="550e8400-e29b-41d4-a716-446655440000",
            share_url="https://api.example.com/api/shared/770e8400-e29b-41d4-a716-446655440222",
            created_at=now,
            expires_at=expires,
            access_count=0,
            last_accessed_at=None
        )
        
        assert schema.share_id == "770e8400-e29b-41d4-a716-446655440222"
        assert schema.session_id == "550e8400-e29b-41d4-a716-446655440000"
        assert schema.access_count == 0
        assert schema.last_accessed_at is None
    
    def test_accessed_share_link(self):
        """Test share link with access history"""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=24)
        last_accessed = now + timedelta(hours=1)
        
        schema = ShareLinkResponseSchema(
            share_id="770e8400-e29b-41d4-a716-446655440222",
            session_id="550e8400-e29b-41d4-a716-446655440000",
            share_url="https://api.example.com/api/shared/770e8400-e29b-41d4-a716-446655440222",
            created_at=now,
            expires_at=expires,
            access_count=5,
            last_accessed_at=last_accessed
        )
        
        assert schema.access_count == 5
        assert schema.last_accessed_at == last_accessed


class TestSharedSessionResponseSchema:
    """Tests for SharedSessionResponseSchema"""
    
    def test_valid_shared_session(self):
        """Test valid shared session response"""
        now = datetime.now(timezone.utc)
        
        schema = SharedSessionResponseSchema(
            id="550e8400-e29b-41d4-a716-446655440000",
            name="Shared Project",
            created_at=now,
            prediction_count=10
        )
        
        assert schema.id == "550e8400-e29b-41d4-a716-446655440000"
        assert schema.name == "Shared Project"
        assert schema.prediction_count == 10
    
    def test_no_user_id_in_shared_session(self):
        """Test that shared session doesn't expose user_id"""
        now = datetime.now(timezone.utc)
        
        schema = SharedSessionResponseSchema(
            id="550e8400-e29b-41d4-a716-446655440000",
            name="Shared Project",
            created_at=now,
            prediction_count=10
        )
        
        # Verify user_id is not in the schema
        json_data = schema.model_dump()
        assert "user_id" not in json_data
    
    def test_default_prediction_count(self):
        """Test default prediction count"""
        now = datetime.now(timezone.utc)
        
        schema = SharedSessionResponseSchema(
            id="550e8400-e29b-41d4-a716-446655440000",
            name="Shared Project",
            created_at=now
        )
        
        assert schema.prediction_count == 0


class TestSchemaExamples:
    """Tests for schema examples in documentation"""
    
    def test_work_session_create_example(self):
        """Test that example from WorkSessionCreateSchema is valid"""
        example = {
            "name": "Antibody Design Project"
        }
        schema = WorkSessionCreateSchema(**example)
        assert schema.name == "Antibody Design Project"
    
    def test_share_link_create_example(self):
        """Test that example from ShareLinkCreateSchema is valid"""
        example = {
            "expiration_hours": 24
        }
        schema = ShareLinkCreateSchema(**example)
        assert schema.expiration_hours == 24


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""
    
    def test_session_name_exactly_255_chars(self):
        """Test session name with exactly 255 characters"""
        name = "A" * 255
        schema = WorkSessionCreateSchema(name=name)
        assert len(schema.name) == 255
    
    def test_session_name_254_chars(self):
        """Test session name with 254 characters"""
        name = "A" * 254
        schema = WorkSessionCreateSchema(name=name)
        assert len(schema.name) == 254
    
    def test_expiration_boundary_values(self):
        """Test expiration hours at boundary values"""
        # Minimum boundary
        schema = ShareLinkCreateSchema(expiration_hours=1)
        assert schema.expiration_hours == 1
        
        # Maximum boundary
        schema = ShareLinkCreateSchema(expiration_hours=168)
        assert schema.expiration_hours == 168
        
        # Just inside boundaries
        schema = ShareLinkCreateSchema(expiration_hours=2)
        assert schema.expiration_hours == 2
        
        schema = ShareLinkCreateSchema(expiration_hours=167)
        assert schema.expiration_hours == 167
    
    def test_unicode_session_names(self):
        """Test session names with unicode characters"""
        # Emoji
        schema = WorkSessionCreateSchema(name="Project 🧬 DNA")
        assert "🧬" in schema.name
        
        # Non-ASCII characters
        schema = WorkSessionCreateSchema(name="Protéine étude")
        assert schema.name == "Protéine étude"
        
        # Asian characters
        schema = WorkSessionCreateSchema(name="タンパク質")
        assert schema.name == "タンパク質"
