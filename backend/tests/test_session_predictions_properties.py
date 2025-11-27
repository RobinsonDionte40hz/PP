"""
Property tests for session predictions functionality
"""
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone

from app.database import Base
from app.models.work_session import WorkSession
from app.models.prediction import Prediction, PredictionStatus
from app.services.work_session_service import WorkSessionService

# Create test database
TEST_DATABASE_URL = "sqlite:///./test_session_predictions_properties.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)


@pytest.fixture(autouse=True)
def setup_database():
    """Clear database before each test"""
    with TestingSessionLocal() as db:
        try:
            db.execute(text("DELETE FROM predictions"))
        except:
            pass
        try:
            db.execute(text("DELETE FROM work_sessions"))
        except:
            pass
        db.commit()
    
    yield
    
    # Clear after test
    with TestingSessionLocal() as db:
        try:
            db.execute(text("DELETE FROM predictions"))
        except:
            pass
        try:
            db.execute(text("DELETE FROM work_sessions"))
        except:
            pass
        db.commit()


@pytest.fixture(scope="module", autouse=True)
def cleanup_test_db():
    """Remove test database file after all tests"""
    yield
    engine.dispose()
    import os
    import time
    if os.path.exists("./test_session_predictions_properties.db"):
        time.sleep(0.1)
        try:
            os.remove("./test_session_predictions_properties.db")
        except PermissionError:
            pass


class TestProperty6_PredictionsLinkedToSessions:
    """
    Property 6: Predictions are linked to sessions
    Validates: Requirements 2.1
    """
    
    def test_prediction_linked_to_session(self):
        """Test that predictions can be linked to sessions"""
        db = TestingSessionLocal()
        service = WorkSessionService(db=db)
        
        try:
            # Create session
            session = service.create_session(
                user_id="test_user_1",
                name="Test Session"
            )
            
            # Create prediction
            prediction = Prediction(
                id="pred_test123",
                sequence="ACDEFGH",
                status=PredictionStatus.PENDING.value,
                configuration={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                session_id=None  # Not linked initially
            )
            db.add(prediction)
            db.commit()
            db.refresh(prediction)
            
            # Link prediction to session
            success = service.create_prediction_in_session(
                session_id=session.id,  # type: ignore[arg-type]
                user_id="test_user_1",
                prediction=prediction
            )
            
            # Verify linking succeeded
            assert success is True
            
            # Verify prediction is linked
            db.refresh(prediction)
            assert prediction.session_id == session.id
            
            # Verify prediction appears in session
            predictions, total = service.get_session_predictions(
                session_id=session.id,  # type: ignore[arg-type]
                user_id="test_user_1"
            )
            
            assert total == 1
            assert predictions[0].id == "pred_test123"
        
        finally:
            db.close()
    
    def test_multiple_predictions_in_session(self):
        """Test that multiple predictions can be linked to same session"""
        db = TestingSessionLocal()
        service = WorkSessionService(db=db)
        
        try:
            # Create session
            session = service.create_session(
                user_id="test_user_1",
                name="Multi-Prediction Session"
            )
            
            # Create and link 5 predictions
            for i in range(5):
                prediction = Prediction(
                    id=f"pred_test{i}",
                    sequence=f"ACDEFG{i}",
                    status=PredictionStatus.PENDING.value,
                    configuration={},
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                    session_id=None
                )
                db.add(prediction)
                db.commit()
                db.refresh(prediction)
                
                service.create_prediction_in_session(
                    session_id=session.id,  # type: ignore[arg-type]
                    user_id="test_user_1",
                    prediction=prediction
                )
            
            # Verify all predictions are linked
            predictions, total = service.get_session_predictions(
                session_id=session.id,  # type: ignore[arg-type]
                user_id="test_user_1"
            )
            
            assert total == 5
            assert len(predictions) == 5
            assert all(p.session_id == session.id for p in predictions)
        
        finally:
            db.close()


class TestProperty8_SessionPredictionsRetrievable:
    """
    Property 8: Session predictions are retrievable
    Validates: Requirements 2.3
    """
    
    def test_predictions_retrievable_by_session(self):
        """Test that predictions can be retrieved by session ID"""
        db = TestingSessionLocal()
        service = WorkSessionService(db=db)
        
        try:
            # Create session
            session = service.create_session(
                user_id="test_user_1",
                name="Retrieval Test Session"
            )
            
            # Create and link predictions
            prediction_ids = []
            for i in range(3):
                prediction = Prediction(
                    id=f"pred_retrieve{i}",
                    sequence=f"ACDEFGH{i}",
                    status=PredictionStatus.PENDING.value,
                    configuration={},
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                    session_id=None
                )
                db.add(prediction)
                db.commit()
                db.refresh(prediction)
                
                service.create_prediction_in_session(
                    session_id=session.id,  # type: ignore[arg-type]
                    user_id="test_user_1",
                    prediction=prediction
                )
                prediction_ids.append(prediction.id)
            
            # Retrieve predictions
            predictions, total = service.get_session_predictions(
                session_id=session.id,  # type: ignore[arg-type]
                user_id="test_user_1"
            )
            
            # Verify all predictions are retrieved
            assert total == 3
            retrieved_ids = [p.id for p in predictions]
            assert set(retrieved_ids) == set(prediction_ids)
        
        finally:
            db.close()
    
    def test_predictions_ordered_by_created_at(self):
        """Test that predictions are ordered by created_at descending"""
        db = TestingSessionLocal()
        service = WorkSessionService(db=db)
        
        try:
            # Create session
            session = service.create_session(
                user_id="test_user_1",
                name="Order Test Session"
            )
            
            # Create predictions with different timestamps
            import time
            for i in range(3):
                prediction = Prediction(
                    id=f"pred_order{i}",
                    sequence=f"ACDEFGH{i}",
                    status=PredictionStatus.PENDING.value,
                    configuration={},
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                    session_id=None
                )
                db.add(prediction)
                db.commit()
                db.refresh(prediction)
                
                service.create_prediction_in_session(
                    session_id=session.id,  # type: ignore[arg-type]
                    user_id="test_user_1",
                    prediction=prediction
                )
                
                time.sleep(0.01)  # Small delay to ensure different timestamps
            
            # Retrieve predictions
            predictions, _ = service.get_session_predictions(
                session_id=session.id,  # type: ignore[arg-type]
                user_id="test_user_1"
            )
            
            # Verify ordering (most recent first)
            assert predictions[0].id == "pred_order2"
            assert predictions[1].id == "pred_order1"
            assert predictions[2].id == "pred_order0"
        
        finally:
            db.close()
    
    def test_predictions_paginated(self):
        """Test that prediction retrieval supports pagination"""
        db = TestingSessionLocal()
        service = WorkSessionService(db=db)
        
        try:
            # Create session
            session = service.create_session(
                user_id="test_user_1",
                name="Pagination Test Session"
            )
            
            # Create 10 predictions
            for i in range(10):
                prediction = Prediction(
                    id=f"pred_page{i}",
                    sequence=f"ACDEFGH{i}",
                    status=PredictionStatus.PENDING.value,
                    configuration={},
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                    session_id=None
                )
                db.add(prediction)
                db.commit()
                db.refresh(prediction)
                
                service.create_prediction_in_session(
                    session_id=session.id,  # type: ignore[arg-type]
                    user_id="test_user_1",
                    prediction=prediction
                )
            
            # Get first page
            page1_predictions, total = service.get_session_predictions(
                session_id=session.id,  # type: ignore[arg-type]
                user_id="test_user_1",
                page=1,
                page_size=3
            )
            
            assert total == 10
            assert len(page1_predictions) == 3
            
            # Get second page
            page2_predictions, total = service.get_session_predictions(
                session_id=session.id,  # type: ignore[arg-type]
                user_id="test_user_1",
                page=2,
                page_size=3
            )
            
            assert total == 10
            assert len(page2_predictions) == 3
            
            # Verify pages have different predictions
            page1_ids = [p.id for p in page1_predictions]
            page2_ids = [p.id for p in page2_predictions]
            assert set(page1_ids).isdisjoint(set(page2_ids))
        
        finally:
            db.close()
    
    def test_empty_session_returns_empty_list(self):
        """Test that empty sessions return empty prediction list"""
        db = TestingSessionLocal()
        service = WorkSessionService(db=db)
        
        try:
            # Create session
            session = service.create_session(
                user_id="test_user_1",
                name="Empty Session"
            )
            
            # Retrieve predictions
            predictions, total = service.get_session_predictions(
                session_id=session.id,  # type: ignore[arg-type]
                user_id="test_user_1"
            )
            
            assert total == 0
            assert predictions == []
        
        finally:
            db.close()


class TestProperty5_SessionActivityUpdatesTimestamp:
    """
    Property 5: Session activity updates timestamp
    Validates: Requirements 1.5, 10.5
    """
    
    def test_linking_prediction_updates_activity(self):
        """Test that linking a prediction updates session activity timestamp"""
        db = TestingSessionLocal()
        service = WorkSessionService(db=db)
        
        try:
            # Create session
            session = service.create_session(
                user_id="test_user_1",
                name="Activity Test Session"
            )
            
            initial_activity = session.last_active_at
            
            # Wait a moment
            import time
            time.sleep(0.01)
            
            # Create and link prediction
            prediction = Prediction(
                id="pred_activity",
                sequence="ACDEFGH",
                status=PredictionStatus.PENDING.value,
                configuration={},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                session_id=None
            )
            db.add(prediction)
            db.commit()
            db.refresh(prediction)
            
            service.create_prediction_in_session(
                session_id=session.id,  # type: ignore[arg-type]
                user_id="test_user_1",
                prediction=prediction
            )
            
            # Verify activity timestamp was updated
            db.refresh(session)
            assert session.last_active_at > initial_activity
        
        finally:
            db.close()
