"""
Unit tests for PhaseManager

Tests:
- Phase creation and validation
- Phase initialization with protein distribution
- Difficulty-based sorting
- Phase status transitions
- Quality gate checking
- Phase summary generation
- Parameter adjustment recommendations
- Export/import functionality
- Edge cases and error handling
"""

import pytest
import json
import os
import tempfile
from datetime import datetime
from typing import List

from validation.phase_manager import (
    PhaseManager,
    Phase,
    PhaseStatus,
    QualityGateResult,
    PhaseSummaryReport
)
from validation.protein_selector import ProteinSelector, ProteinMetadata


class TestPhase:
    """Test Phase dataclass."""
    
    def test_valid_phase(self):
        """Test creating valid Phase."""
        proteins = []  # Empty for testing
        
        phase = Phase(
            phase_number=1,
            protein_count=10,
            proteins=proteins,
            status=PhaseStatus.PENDING
        )
        
        assert phase.phase_number == 1
        assert phase.protein_count == 10
        assert phase.status == PhaseStatus.PENDING
        assert not phase.is_complete()
    
    def test_invalid_phase_number(self):
        """Test that invalid phase number raises ValueError."""
        with pytest.raises(ValueError, match="phase_number must be 1-4"):
            Phase(
                phase_number=5,  # Invalid
                protein_count=10,
                proteins=[],
                status=PhaseStatus.PENDING
            )
    
    def test_phase_status_conversion(self):
        """Test status string to enum conversion."""
        phase = Phase(
            phase_number=1,
            protein_count=10,
            proteins=[],
            status="pending"  # String instead of enum
        )
        
        assert isinstance(phase.status, PhaseStatus)
        assert phase.status == PhaseStatus.PENDING
    
    def test_phase_completion_check(self):
        """Test phase completion checking."""
        phase = Phase(
            phase_number=1,
            protein_count=10,
            proteins=[],
            status=PhaseStatus.IN_PROGRESS
        )
        
        assert not phase.is_complete()
        
        phase.status = PhaseStatus.COMPLETED
        assert phase.is_complete()
        
        phase.status = PhaseStatus.FAILED_GATE
        assert phase.is_complete()
    
    def test_phase_duration_calculation(self):
        """Test phase duration calculation."""
        phase = Phase(
            phase_number=1,
            protein_count=10,
            proteins=[],
            status=PhaseStatus.IN_PROGRESS
        )
        
        # No duration initially
        assert phase.get_duration_seconds() is None
        
        # Set times
        phase.start_time = datetime(2025, 10, 26, 10, 0, 0)
        phase.end_time = datetime(2025, 10, 26, 10, 30, 0)
        
        # Should be 30 minutes = 1800 seconds
        assert phase.get_duration_seconds() == 1800.0


class TestQualityGateResult:
    """Test QualityGateResult dataclass."""
    
    def test_quality_gate_passed(self):
        """Test quality gate that passed."""
        result = QualityGateResult(
            passed=True,
            success_rate=75.0,
            threshold=60.0,
            issues_identified=[],
            recommendations=["Continue to next phase"],
            phase_number=1
        )
        
        assert result.passed
        assert result.success_rate == 75.0
        assert len(result.issues_identified) == 0
    
    def test_quality_gate_failed(self):
        """Test quality gate that failed."""
        result = QualityGateResult(
            passed=False,
            success_rate=45.0,
            threshold=60.0,
            issues_identified=["Low success rate", "High RMSD"],
            recommendations=["Increase agent count"],
            phase_number=1
        )
        
        assert not result.passed
        assert len(result.issues_identified) == 2
        assert len(result.recommendations) == 1
    
    def test_quality_gate_summary(self):
        """Test quality gate summary generation."""
        result = QualityGateResult(
            passed=True,
            success_rate=70.0,
            threshold=60.0,
            issues_identified=[],
            recommendations=["Continue"],
            phase_number=1
        )
        
        summary = result.get_summary()
        assert "Phase 1" in summary
        assert "PASSED" in summary
        assert "70.0%" in summary


class TestPhaseManager:
    """Test PhaseManager functionality."""
    
    @pytest.fixture
    def selector(self):
        """Create ProteinSelector for testing."""
        return ProteinSelector()
    
    @pytest.fixture
    def proteins(self, selector):
        """Get test proteins."""
        return selector.select_proteins(target_count=30)
    
    @pytest.fixture
    def manager(self):
        """Create PhaseManager for testing."""
        return PhaseManager(
            phase1_count=5,
            phase2_count=8,
            phase3_count=12,
            quality_gate_threshold=60.0
        )
    
    def test_manager_initialization(self, manager):
        """Test manager initializes correctly."""
        assert manager.phase1_count == 5
        assert manager.phase2_count == 8
        assert manager.phase3_count == 12
        assert manager.quality_gate_threshold == 60.0
        assert manager.current_phase_number == 1
        assert len(manager.phases) == 0  # Not initialized yet
    
    def test_initialize_phases(self, manager, proteins):
        """Test phase initialization with proteins."""
        phases = manager.initialize_phases(proteins)
        
        assert len(phases) == 4
        assert all(i in phases for i in [1, 2, 3, 4])
        
        # Check phase protein counts
        total_proteins = sum(phase.protein_count for phase in phases.values())
        assert total_proteins == len(proteins)
    
    def test_phase_protein_distribution(self, manager, proteins):
        """Test proteins are distributed correctly across phases."""
        phases = manager.initialize_phases(proteins)
        
        # Each phase should have proteins
        for phase_num, phase in phases.items():
            assert len(phase.proteins) == phase.protein_count
            assert all(isinstance(p, ProteinMetadata) for p in phase.proteins)
    
    def test_difficulty_sorting(self, manager, proteins):
        """Test proteins are sorted by difficulty."""
        sorted_proteins = manager._sort_by_difficulty(proteins)
        
        # First protein should be easier (small size, good resolution)
        # Last protein should be harder (large size or poor resolution)
        first = sorted_proteins[0]
        last = sorted_proteins[-1]
        
        # Generally, first should be smaller
        assert first.sequence_length <= last.sequence_length * 1.5  # Allow some variance
    
    def test_get_current_phase(self, manager, proteins):
        """Test getting current phase."""
        manager.initialize_phases(proteins)
        
        current = manager.get_current_phase()
        assert current.phase_number == 1
        assert current.status == PhaseStatus.PENDING
    
    def test_get_current_phase_before_init(self, manager):
        """Test getting current phase before initialization raises error."""
        with pytest.raises(ValueError, match="No phases initialized"):
            manager.get_current_phase()
    
    def test_get_phase_by_number(self, manager, proteins):
        """Test getting specific phase by number."""
        manager.initialize_phases(proteins)
        
        phase2 = manager.get_phase(2)
        assert phase2.phase_number == 2
        
        phase4 = manager.get_phase(4)
        assert phase4.phase_number == 4
    
    def test_get_invalid_phase(self, manager, proteins):
        """Test getting invalid phase raises error."""
        manager.initialize_phases(proteins)
        
        with pytest.raises(ValueError, match="does not exist"):
            manager.get_phase(5)
    
    def test_phase_transitions(self, manager, proteins):
        """Test phase status transitions."""
        manager.initialize_phases(proteins)
        
        # Start phase 1
        manager.start_phase(1)
        phase1 = manager.get_phase(1)
        assert phase1.status == PhaseStatus.IN_PROGRESS
        assert phase1.start_time is not None
        
        # Complete phase 1
        manager.complete_phase(1)
        assert phase1.status == PhaseStatus.COMPLETED
        assert phase1.end_time is not None
    
    def test_advance_to_next_phase(self, manager, proteins):
        """Test advancing to next phase."""
        manager.initialize_phases(proteins)
        
        # Complete phase 1
        manager.start_phase(1)
        manager.complete_phase(1)
        
        # Advance to phase 2
        advanced = manager.advance_to_next_phase()
        assert advanced
        assert manager.current_phase_number == 2
        
        # Current phase should now be phase 2
        current = manager.get_current_phase()
        assert current.phase_number == 2
    
    def test_advance_without_completing(self, manager, proteins):
        """Test advancing without completing current phase raises error."""
        manager.initialize_phases(proteins)
        
        # Try to advance without completing
        with pytest.raises(ValueError, match="is not complete"):
            manager.advance_to_next_phase()
    
    def test_advance_from_last_phase(self, manager, proteins):
        """Test advancing from last phase returns False."""
        manager.initialize_phases(proteins)
        
        # Complete all phases
        for i in range(1, 5):
            manager.start_phase(i)
            manager.complete_phase(i)
            if i < 4:
                manager.advance_to_next_phase()
        
        # Try to advance from phase 4
        advanced = manager.advance_to_next_phase()
        assert not advanced
        assert manager.current_phase_number == 4
    
    def test_update_phase_results(self, manager, proteins):
        """Test updating phase with results."""
        manager.initialize_phases(proteins)
        phase = manager.get_current_phase()
        
        # Mock results
        results = [
            {'pdb_id': '1UBQ', 'success': True, 'rmsd': 2.5, 'gdt_ts': 75.0, 'tm_score': 0.8, 'energy': -50.0, 'execution_time': 120.0},
            {'pdb_id': '1CRN', 'success': True, 'rmsd': 3.0, 'gdt_ts': 70.0, 'tm_score': 0.75, 'energy': -45.0, 'execution_time': 90.0},
            {'pdb_id': '1LYZ', 'success': False, 'rmsd': 8.0, 'gdt_ts': 30.0, 'tm_score': 0.3, 'energy': 10.0, 'execution_time': 150.0},
        ]
        
        manager.update_phase_results(phase, results)
        
        # Check metrics
        assert phase.success_rate == pytest.approx(66.67, rel=0.1)  # 2/3 success
        assert phase.average_rmsd == 2.75  # Average of successful only
        assert len(phase.failed_proteins) == 1
        assert '1LYZ' in phase.failed_proteins
    
    def test_quality_gate_passing(self, manager, proteins):
        """Test quality gate with passing results."""
        manager.initialize_phases(proteins)
        phase = manager.get_current_phase()
        
        # Good results (70% success)
        results = [
            {'pdb_id': f'TEST{i}', 'success': i < 7, 'rmsd': 2.0, 'gdt_ts': 80.0, 
             'tm_score': 0.85, 'energy': -60.0, 'execution_time': 100.0}
            for i in range(10)
        ]
        
        manager.update_phase_results(phase, results)
        gate_result = manager.check_quality_gate(phase)
        
        assert gate_result.passed
        assert gate_result.success_rate == 70.0
        assert len(gate_result.issues_identified) == 0
    
    def test_quality_gate_failing(self, manager, proteins):
        """Test quality gate with failing results."""
        manager.initialize_phases(proteins)
        phase = manager.get_current_phase()
        
        # Poor results (40% success)
        results = [
            {'pdb_id': f'TEST{i}', 'success': i < 4, 'rmsd': 6.0, 'gdt_ts': 40.0,
             'tm_score': 0.4, 'energy': 5.0, 'execution_time': 100.0}
            for i in range(10)
        ]
        
        manager.update_phase_results(phase, results)
        gate_result = manager.check_quality_gate(phase)
        
        assert not gate_result.passed
        assert gate_result.success_rate == 40.0
        assert len(gate_result.issues_identified) > 0
        assert len(gate_result.recommendations) > 0
    
    def test_generate_phase_summary(self, manager, proteins):
        """Test phase summary generation."""
        manager.initialize_phases(proteins)
        phase = manager.get_current_phase()
        
        # Set phase times
        manager.start_phase(1)
        phase.start_time = datetime.now()
        
        # Mock results
        results = [
            {'pdb_id': '1UBQ', 'success': True, 'rmsd': 2.0, 'gdt_ts': 80.0, 'tm_score': 0.85, 'energy': -60.0, 'execution_time': 100.0},
            {'pdb_id': '1CRN', 'success': True, 'rmsd': 2.5, 'gdt_ts': 75.0, 'tm_score': 0.80, 'energy': -55.0, 'execution_time': 90.0},
            {'pdb_id': '1LYZ', 'success': True, 'rmsd': 3.0, 'gdt_ts': 70.0, 'tm_score': 0.75, 'energy': -50.0, 'execution_time': 120.0},
        ]
        
        manager.update_phase_results(phase, results)
        manager.complete_phase(1)
        
        summary = manager.generate_phase_summary(phase, results)
        
        assert summary.phase_number == 1
        assert summary.proteins_tested == 3
        assert summary.proteins_succeeded == 3
        assert summary.proteins_failed == 0
        assert summary.success_rate == 100.0
        assert len(summary.top_performers) > 0
    
    def test_parameter_adjustment_good_performance(self, manager, proteins):
        """Test parameter adjustment with good performance."""
        manager.initialize_phases(proteins)
        phase = manager.get_current_phase()
        
        # Good performance
        phase.success_rate = 80.0
        phase.average_rmsd = 2.5
        phase.average_gdt_ts = 75.0
        phase.average_energy = -50.0
        
        adjustments = manager.allow_parameter_adjustment(phase)
        
        # Should recommend no changes
        assert 'status' in adjustments
        assert adjustments['status'] == 'no_changes_needed'
    
    def test_parameter_adjustment_poor_performance(self, manager, proteins):
        """Test parameter adjustment with poor performance."""
        manager.initialize_phases(proteins)
        phase = manager.get_current_phase()
        
        # Poor performance
        phase.success_rate = 50.0
        phase.average_rmsd = 6.0
        phase.average_gdt_ts = 45.0
        phase.average_energy = 10.0
        
        adjustments = manager.allow_parameter_adjustment(phase)
        
        # Should recommend changes
        assert 'num_agents' in adjustments
        assert 'iterations' in adjustments
        assert 'exploration' in adjustments
    
    def test_export_import_phases(self, manager, proteins):
        """Test exporting and importing phases."""
        # Initialize phases
        manager.initialize_phases(proteins)
        manager.start_phase(1)
        
        # Export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            manager.export_phases(json_path)
            assert os.path.exists(json_path)
            
            # Create new manager and import
            new_manager = PhaseManager()
            new_manager.load_phases(json_path)
            
            # Verify
            assert new_manager.phase1_count == manager.phase1_count
            assert new_manager.current_phase_number == manager.current_phase_number
            assert len(new_manager.phases) == len(manager.phases)
            
            # Check phase details
            for phase_num in [1, 2, 3, 4]:
                original_phase = manager.get_phase(phase_num)
                loaded_phase = new_manager.get_phase(phase_num)
                assert original_phase.phase_number == loaded_phase.phase_number
                assert original_phase.protein_count == loaded_phase.protein_count
                assert original_phase.status == loaded_phase.status
        
        finally:
            if os.path.exists(json_path):
                os.unlink(json_path)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_protein_list(self):
        """Test initializing with empty protein list."""
        manager = PhaseManager(phase1_count=5, phase2_count=5, phase3_count=5)
        
        # Should handle empty list gracefully
        phases = manager.initialize_phases([])
        
        # All phases should exist but be empty
        assert len(phases) == 4
        for phase in phases.values():
            assert phase.protein_count == 0
    
    def test_insufficient_proteins(self):
        """Test with fewer proteins than requested."""
        selector = ProteinSelector()
        proteins = selector.select_proteins(target_count=10)
        
        manager = PhaseManager(phase1_count=5, phase2_count=10, phase3_count=15)
        
        # Should adjust and distribute available proteins
        phases = manager.initialize_phases(proteins)
        
        total_distributed = sum(p.protein_count for p in phases.values())
        assert total_distributed == len(proteins)
    
    def test_update_phase_with_empty_results(self, ):
        """Test updating phase with empty results list."""
        selector = ProteinSelector()
        proteins = selector.select_proteins(target_count=20)
        
        manager = PhaseManager()
        manager.initialize_phases(proteins)
        phase = manager.get_current_phase()
        
        # Update with empty results
        manager.update_phase_results(phase, [])
        
        # Should not crash, metrics should remain at defaults
        assert phase.success_rate == 0.0
        assert phase.average_rmsd == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
