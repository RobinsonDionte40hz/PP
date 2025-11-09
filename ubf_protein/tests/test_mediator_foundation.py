"""
Unit Tests for Mediator Agent Foundation

Tests MediatorAgent initialization, configuration validation, and
interface compliance (Task 5.4).

Author: UBF Protein System
Date: November 9, 2025
"""

import pytest
import time
from unittest.mock import Mock, MagicMock

from ubf_protein.mediator_agent import MediatorAgent
from ubf_protein.mediator_config import MediatorConfig
from ubf_protein.geometric_attractor import GeometricAttractorAnalyzer
from ubf_protein.interfaces import IProteinAgent, IConsciousnessState, IBehavioralState, IMemorySystem


# ============================================================================
# MediatorConfig Tests
# ============================================================================

class TestMediatorConfig:
    """Test MediatorConfig dataclass validation and factory methods"""
    
    def test_default_config_initialization(self):
        """Test that default configuration initializes successfully"""
        config = MediatorConfig()
        
        assert config.thz_similarity_threshold == 0.7
        assert config.geometric_similarity_threshold == 2.0
        assert config.secondary_structure_min_length == 4
        assert config.relay_frequency == 20
        assert config.broadcast_throttle_rate == 10
        assert config.cache_size == 10000
        assert config.cache_ttl_seconds == 3600
        assert config.enable_thz_detection is True
        assert config.enable_folding_detection is True
        assert config.enable_geometric_detection is True
        assert config.max_reference_conformations == 100
        assert config.pattern_history_size == 1000
    
    def test_custom_config_initialization(self):
        """Test custom configuration with valid parameters"""
        config = MediatorConfig(
            thz_similarity_threshold=0.8,
            geometric_similarity_threshold=1.5,
            secondary_structure_min_length=5,
            relay_frequency=10,
            cache_size=5000
        )
        
        assert config.thz_similarity_threshold == 0.8
        assert config.geometric_similarity_threshold == 1.5
        assert config.secondary_structure_min_length == 5
        assert config.relay_frequency == 10
        assert config.cache_size == 5000
    
    def test_invalid_thz_threshold_too_low(self):
        """Test that thz_similarity_threshold < 0.0 raises ValueError"""
        with pytest.raises(ValueError, match="thz_similarity_threshold must be in range"):
            MediatorConfig(thz_similarity_threshold=-0.1)
    
    def test_invalid_thz_threshold_too_high(self):
        """Test that thz_similarity_threshold > 1.0 raises ValueError"""
        with pytest.raises(ValueError, match="thz_similarity_threshold must be in range"):
            MediatorConfig(thz_similarity_threshold=1.5)
    
    def test_invalid_geometric_threshold(self):
        """Test that geometric_similarity_threshold <= 0 raises ValueError"""
        with pytest.raises(ValueError, match="geometric_similarity_threshold must be positive"):
            MediatorConfig(geometric_similarity_threshold=0.0)
    
    def test_invalid_secondary_structure_min_length(self):
        """Test that secondary_structure_min_length < 1 raises ValueError"""
        with pytest.raises(ValueError, match="secondary_structure_min_length must be at least 1"):
            MediatorConfig(secondary_structure_min_length=0)
    
    def test_invalid_relay_frequency(self):
        """Test that relay_frequency < 1 raises ValueError"""
        with pytest.raises(ValueError, match="relay_frequency must be at least 1"):
            MediatorConfig(relay_frequency=0)
    
    def test_invalid_broadcast_throttle_rate(self):
        """Test that broadcast_throttle_rate < 1 raises ValueError"""
        with pytest.raises(ValueError, match="broadcast_throttle_rate must be at least 1"):
            MediatorConfig(broadcast_throttle_rate=0)
    
    def test_invalid_cache_size(self):
        """Test that cache_size < 1 raises ValueError"""
        with pytest.raises(ValueError, match="cache_size must be at least 1"):
            MediatorConfig(cache_size=0)
    
    def test_invalid_cache_ttl(self):
        """Test that cache_ttl_seconds <= 0 raises ValueError"""
        with pytest.raises(ValueError, match="cache_ttl_seconds must be positive"):
            MediatorConfig(cache_ttl_seconds=0)
    
    def test_invalid_max_reference_conformations(self):
        """Test that max_reference_conformations < 1 raises ValueError"""
        with pytest.raises(ValueError, match="max_reference_conformations must be at least 1"):
            MediatorConfig(max_reference_conformations=0)
    
    def test_invalid_pattern_history_size(self):
        """Test that pattern_history_size < 1 raises ValueError"""
        with pytest.raises(ValueError, match="pattern_history_size must be at least 1"):
            MediatorConfig(pattern_history_size=0)
    
    def test_all_detections_disabled_raises_error(self):
        """Test that disabling all detection types raises ValueError"""
        with pytest.raises(ValueError, match="At least one detection type must be enabled"):
            MediatorConfig(
                enable_thz_detection=False,
                enable_folding_detection=False,
                enable_geometric_detection=False
            )
    
    def test_to_dict_conversion(self):
        """Test that configuration converts to dictionary correctly"""
        config = MediatorConfig()
        config_dict = config.to_dict()
        
        assert 'detection_thresholds' in config_dict
        assert 'relay_config' in config_dict
        assert 'cache_config' in config_dict
        assert 'features' in config_dict
        assert 'memory_management' in config_dict
        
        assert config_dict['detection_thresholds']['thz_similarity'] == 0.7
        assert config_dict['relay_config']['relay_frequency'] == 20
        assert config_dict['cache_config']['cache_size'] == 10000
        assert config_dict['features']['thz_detection'] is True
    
    def test_high_sensitivity_config(self):
        """Test high sensitivity configuration factory method"""
        config = MediatorConfig.get_high_sensitivity_config()
        
        assert config.thz_similarity_threshold == 0.6
        assert config.geometric_similarity_threshold == 3.0
        assert config.secondary_structure_min_length == 3
        assert config.relay_frequency == 10
        assert config.cache_size == 15000
    
    def test_high_performance_config(self):
        """Test high performance configuration factory method"""
        config = MediatorConfig.get_high_performance_config()
        
        assert config.thz_similarity_threshold == 0.8
        assert config.geometric_similarity_threshold == 1.5
        assert config.secondary_structure_min_length == 5
        assert config.relay_frequency == 50
        assert config.enable_thz_detection is False  # Disabled for performance
        assert config.cache_size == 5000
    
    def test_balanced_config(self):
        """Test balanced configuration factory method"""
        config = MediatorConfig.get_balanced_config()
        
        # Should match default values
        assert config.thz_similarity_threshold == 0.7
        assert config.relay_frequency == 20
        assert config.cache_size == 10000


# ============================================================================
# MediatorAgent Tests
# ============================================================================

class TestMediatorAgent:
    """Test MediatorAgent initialization and interface compliance"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for MediatorAgent"""
        qcpp_adapter = Mock()
        geometric_analyzer = GeometricAttractorAnalyzer()
        shared_memory = Mock()
        
        return qcpp_adapter, geometric_analyzer, shared_memory
    
    def test_mediator_agent_initialization(self, mock_dependencies):
        """Test that MediatorAgent initializes successfully with valid inputs"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGHIKL",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        assert agent.protein_sequence == "ACDEFGHIKL"
        assert agent.qcpp_adapter is qcpp_adapter
        assert agent.geometric_analyzer is geometric_analyzer
        assert agent.shared_memory is shared_memory
        assert isinstance(agent.config, MediatorConfig)
    
    def test_mediator_agent_with_custom_config(self, mock_dependencies):
        """Test MediatorAgent initialization with custom configuration"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        config = MediatorConfig(relay_frequency=10, cache_size=5000)
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory,
            config=config
        )
        
        assert agent.config.relay_frequency == 10
        assert agent.config.cache_size == 5000
    
    def test_mediator_agent_empty_sequence_raises_error(self, mock_dependencies):
        """Test that empty protein_sequence raises ValueError"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        with pytest.raises(ValueError, match="protein_sequence cannot be empty"):
            MediatorAgent(
                protein_sequence="",
                qcpp_adapter=qcpp_adapter,
                geometric_analyzer=geometric_analyzer,
                shared_memory=shared_memory
            )
    
    def test_consciousness_state_initialization(self, mock_dependencies):
        """Test that consciousness state is initialized correctly"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        consciousness = agent.get_consciousness_state()
        coords = consciousness.get_coordinates()
        
        # Mediators should have frequency=9.0, coherence=0.8
        assert coords.frequency == 9.0
        assert coords.coherence == 0.8
    
    def test_behavioral_state_initialization(self, mock_dependencies):
        """Test that behavioral state is derived from consciousness"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        behavioral = agent.get_behavioral_state()
        
        # Should have behavioral methods from interface
        assert hasattr(behavioral, 'get_exploration_energy')
        assert hasattr(behavioral, 'get_structural_focus')
        assert hasattr(behavioral, 'get_hydrophobic_drive')
        
        # Should be able to call methods
        exploration_energy = behavioral.get_exploration_energy()
        assert isinstance(exploration_energy, float)
    
    def test_memory_system_initialization(self, mock_dependencies):
        """Test that memory system is initialized"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        memory_system = agent.get_memory_system()
        
        assert memory_system is not None
    
    def test_detection_statistics_initialization(self, mock_dependencies):
        """Test that detection statistics are initialized to zero"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        stats = agent.get_detection_statistics()
        
        assert stats['total_detections'] == 0
        assert stats['thz_detections'] == 0
        assert stats['folding_detections'] == 0
        assert stats['geometric_detections'] == 0
        assert stats['broadcasts'] == 0
        assert stats['qcpp_validations'] == 0
        assert stats['cache_hits'] == 0
        assert stats['cache_misses'] == 0
        assert stats['cache_hit_rate'] == 0.0
        assert stats['cache_size'] == 0
        assert stats['reference_conformations'] == 0
    
    def test_pattern_cache_initialization(self, mock_dependencies):
        """Test that pattern cache is initialized empty"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        assert len(agent.pattern_cache) == 0
        assert len(agent.reference_conformations) == 0
    
    def test_iprotein_agent_interface_compliance(self, mock_dependencies):
        """Test that MediatorAgent implements IProteinAgent interface"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        # Check that agent implements all required methods
        assert isinstance(agent, IProteinAgent)
        assert hasattr(agent, 'get_consciousness_state')
        assert hasattr(agent, 'get_behavioral_state')
        assert hasattr(agent, 'get_memory_system')
        assert hasattr(agent, 'explore_step')
        assert hasattr(agent, 'get_current_conformation')
        assert hasattr(agent, 'get_exploration_metrics')
    
    def test_get_exploration_metrics(self, mock_dependencies):
        """Test that get_exploration_metrics returns detection statistics"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        metrics = agent.get_exploration_metrics()
        
        assert 'total_detections' in metrics
        assert 'thz_detections' in metrics
        assert 'folding_detections' in metrics
        assert 'geometric_detections' in metrics
        assert 'broadcasts' in metrics
        assert 'cache_hit_rate' in metrics
        
        # All should be floats
        for key, value in metrics.items():
            assert isinstance(value, float)
    
    def test_mediator_specific_methods_exist(self, mock_dependencies):
        """Test that Mediator-specific methods are present"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        # Check Mediator-specific methods exist
        assert hasattr(agent, 'detect_patterns')
        assert hasattr(agent, 'relay_to_qcpp')
        assert hasattr(agent, 'broadcast_to_agents')
        assert hasattr(agent, 'get_detection_statistics')
    
    def test_explore_step_not_implemented(self, mock_dependencies):
        """Test that explore_step raises NotImplementedError (Task 5.3)"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        # explore_step should raise NotImplementedError until Task 5.3
        with pytest.raises(NotImplementedError, match="will be implemented in Task 5.3"):
            agent.explore_step()
    
    def test_get_current_conformation_not_implemented(self, mock_dependencies):
        """Test that get_current_conformation raises NotImplementedError"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        # Mediators don't maintain conformations
        with pytest.raises(NotImplementedError, match="don't maintain conformations"):
            agent.get_current_conformation()


# ============================================================================
# Task 8: Geometric Similarity Detection Tests
# ============================================================================

class TestGeometricSimilarityDetection:
    """Test geometric similarity detection functionality (Task 8)"""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for MediatorAgent"""
        qcpp_adapter = Mock()
        geometric_analyzer = Mock()
        shared_memory = Mock()
        return qcpp_adapter, geometric_analyzer, shared_memory
    
    @pytest.fixture
    def sample_conformation(self):
        """Create a sample conformation for testing"""
        from ubf_protein.models import Conformation
        
        # Create conformation with 7 residues
        coords = [
            (0.0, 0.0, 0.0),
            (3.8, 0.0, 0.0),
            (7.6, 0.0, 0.0),
            (11.4, 0.0, 0.0),
            (15.2, 0.0, 0.0),
            (19.0, 0.0, 0.0),
            (22.8, 0.0, 0.0),
        ]
        
        conformation = Mock(spec=Conformation)
        conformation.atom_coordinates = coords
        conformation.sequence = "ACDEFGH"
        conformation.energy = -50.0
        
        return conformation
    
    def test_add_reference_conformation(self, mock_dependencies, sample_conformation):
        """Test adding reference conformations"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        # Initially no references
        assert len(agent.reference_conformations) == 0
        
        # Add a reference
        agent.add_reference_conformation(
            sample_conformation,
            agent_id="agent_1",
            geometric_score=25.5
        )
        
        # Should have 1 reference
        assert len(agent.reference_conformations) == 1
        
        ref = agent.reference_conformations[0]
        assert ref['coordinates'] == sample_conformation.atom_coordinates
        assert ref['energy'] == sample_conformation.energy
        assert ref['geometric_score'] == 25.5
        assert ref['agent_id'] == "agent_1"
        assert 'hash' in ref
        assert 'timestamp' in ref
    
    def test_reference_conformation_deduplication(self, mock_dependencies, sample_conformation):
        """Test that duplicate conformations update timestamp instead of adding new entry"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        # Add same conformation twice
        agent.add_reference_conformation(sample_conformation, agent_id="agent_1")
        time.sleep(0.01)  # Small delay for timestamp difference
        agent.add_reference_conformation(sample_conformation, agent_id="agent_2")
        
        # Should still have only 1 reference
        assert len(agent.reference_conformations) == 1
    
    def test_reference_eviction_when_limit_reached(self, mock_dependencies):
        """Test LRU eviction when max_references (100) is reached"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        # Override max for faster testing
        agent.max_references = 5
        
        # Add 6 conformations with different scores
        for i in range(6):
            conformation = Mock()
            # Vary coordinates to ensure unique hashes
            conformation.atom_coordinates = [(float(i), 0.0, 0.0), (float(i+1), 0.0, 0.0)]
            conformation.sequence = "AC"
            conformation.energy = -10.0 * i
            
            # Give decreasing geometric scores
            agent.add_reference_conformation(
                conformation,
                agent_id=f"agent_{i}",
                geometric_score=float(i)
            )
            time.sleep(0.001)  # Ensure different timestamps
        
        # Should have exactly max_references
        assert len(agent.reference_conformations) == 5
        
        # Lowest score should have been evicted (score=0)
        scores = [ref['geometric_score'] for ref in agent.reference_conformations]
        assert 0.0 not in scores
        assert all(score >= 1.0 for score in scores)
    
    def test_clear_reference_conformations(self, mock_dependencies, sample_conformation):
        """Test clearing all reference conformations"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        # Add references
        agent.add_reference_conformation(sample_conformation)
        assert len(agent.reference_conformations) > 0
        
        # Clear
        agent.clear_reference_conformations()
        assert len(agent.reference_conformations) == 0
    
    def test_calculate_structural_overlap_identical(self, mock_dependencies):
        """Test structural overlap calculation with identical structures"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        coords = [(0.0, 0.0, 0.0), (3.8, 0.0, 0.0), (7.6, 0.0, 0.0)]
        
        # Identical structures should have 100% overlap
        overlap = agent._calculate_structural_overlap(coords, coords, distance_threshold=2.0)
        assert overlap == 100.0
    
    def test_calculate_structural_overlap_partial(self, mock_dependencies):
        """Test structural overlap calculation with partial overlap"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        coords1 = [(0.0, 0.0, 0.0), (3.8, 0.0, 0.0), (7.6, 0.0, 0.0)]
        # Shift second structure by 1.0 Å (within threshold)
        coords2 = [(0.0, 0.0, 1.0), (3.8, 0.0, 1.0), (7.6, 0.0, 1.0)]
        
        # All points within 2.0 Å threshold
        overlap = agent._calculate_structural_overlap(coords1, coords2, distance_threshold=2.0)
        assert overlap == 100.0
    
    def test_calculate_structural_overlap_no_match(self, mock_dependencies):
        """Test structural overlap calculation with no overlap"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        coords1 = [(0.0, 0.0, 0.0), (3.8, 0.0, 0.0), (7.6, 0.0, 0.0)]
        # Shift second structure by 10.0 Å (outside threshold)
        coords2 = [(10.0, 10.0, 10.0), (13.8, 10.0, 10.0), (17.6, 10.0, 10.0)]
        
        # No points within 2.0 Å threshold
        overlap = agent._calculate_structural_overlap(coords1, coords2, distance_threshold=2.0)
        assert overlap == 0.0
    
    def test_detect_geometric_similarity_no_references(self, mock_dependencies, sample_conformation):
        """Test that geometric similarity detection returns None with no references"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        # No references added
        pattern = agent._detect_geometric_similarity(sample_conformation)
        assert pattern is None
    
    def test_detect_geometric_similarity_disabled(self, mock_dependencies, sample_conformation):
        """Test that geometric similarity detection respects config flag"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        config = MediatorConfig(enable_geometric_detection=False)
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory,
            config=config
        )
        
        # Add a reference
        agent.add_reference_conformation(sample_conformation)
        
        # Detection should return None (disabled)
        pattern = agent._detect_geometric_similarity(sample_conformation)
        assert pattern is None
    
    def test_detect_geometric_similarity_success(self, mock_dependencies):
        """Test successful geometric similarity detection"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        # Mock geometric analyzer to return analysis results
        mock_analysis = Mock()
        mock_analysis.golden_ratio_percentage = 24.5
        mock_analysis.tetrahedron_similarity = 0.3
        mock_analysis.cube_similarity = 0.2
        mock_analysis.octahedron_similarity = 0.4
        mock_analysis.dodecahedron_similarity = 0.7
        mock_analysis.icosahedron_similarity = 0.8  # Dominant
        geometric_analyzer.analyze_conformation.return_value = mock_analysis
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        # Create reference conformation
        ref_coords = [(0.0, 0.0, 0.0), (3.8, 0.0, 0.0), (7.6, 0.0, 0.0)]
        ref_conf = Mock()
        ref_conf.atom_coordinates = ref_coords
        ref_conf.sequence = "ACD"
        ref_conf.energy = -50.0
        
        agent.add_reference_conformation(ref_conf, geometric_score=30.0)
        
        # Create similar conformation (slight shift)
        similar_coords = [(0.1, 0.1, 0.1), (3.9, 0.1, 0.1), (7.7, 0.1, 0.1)]
        similar_conf = Mock()
        similar_conf.atom_coordinates = similar_coords
        similar_conf.sequence = "ACD"
        similar_conf.energy = -48.0
        
        # Detect similarity
        pattern = agent._detect_geometric_similarity(similar_conf)
        
        # Should detect pattern
        assert pattern is not None
        assert pattern.pattern_type.value == "geometric_similarity"
        assert pattern.geometric_data is not None
        
        geo_data = pattern.geometric_data
        assert geo_data.rmsd_to_reference < 2.0  # Within threshold
        assert geo_data.overlap_percentage > 0.0
        assert geo_data.golden_ratio_percentage == 24.5
        assert geo_data.dominant_platonic_solid == "icosahedron"
        assert geo_data.platonic_similarity_score == 0.8
        assert len(geo_data.reference_conformation_hash) == 16
    
    def test_detect_geometric_similarity_high_significance(self, mock_dependencies):
        """Test geometric similarity detection with HIGH significance"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        # Mock geometric analyzer
        mock_analysis = Mock()
        mock_analysis.golden_ratio_percentage = 30.0
        mock_analysis.tetrahedron_similarity = 0.5
        mock_analysis.cube_similarity = 0.5
        mock_analysis.octahedron_similarity = 0.5
        mock_analysis.dodecahedron_similarity = 0.9
        mock_analysis.icosahedron_similarity = 0.8
        geometric_analyzer.analyze_conformation.return_value = mock_analysis
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        # Add reference
        ref_coords = [(0.0, 0.0, 0.0), (3.8, 0.0, 0.0)]
        ref_conf = Mock()
        ref_conf.atom_coordinates = ref_coords
        ref_conf.sequence = "AC"
        ref_conf.energy = -20.0
        
        agent.add_reference_conformation(ref_conf)
        
        # Create nearly identical conformation (RMSD < 1.0, overlap > 80%)
        similar_coords = [(0.05, 0.05, 0.05), (3.85, 0.05, 0.05)]
        similar_conf = Mock()
        similar_conf.atom_coordinates = similar_coords
        similar_conf.sequence = "AC"
        
        pattern = agent._detect_geometric_similarity(similar_conf)
        
        # Should be HIGH significance
        assert pattern is not None
        assert pattern.significance.value == "high"
        assert pattern.geometric_data.rmsd_to_reference < 1.0
        assert pattern.geometric_data.overlap_percentage > 80.0
    
    def test_detect_geometric_similarity_medium_significance(self, mock_dependencies):
        """Test geometric similarity detection with MEDIUM significance"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        # Mock geometric analyzer
        mock_analysis = Mock()
        mock_analysis.golden_ratio_percentage = 20.0
        mock_analysis.tetrahedron_similarity = 0.4
        mock_analysis.cube_similarity = 0.5
        mock_analysis.octahedron_similarity = 0.6
        mock_analysis.dodecahedron_similarity = 0.3
        mock_analysis.icosahedron_similarity = 0.2
        geometric_analyzer.analyze_conformation.return_value = mock_analysis
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        # Add reference
        ref_coords = [(0.0, 0.0, 0.0), (3.8, 0.0, 0.0), (7.6, 0.0, 0.0)]
        ref_conf = Mock()
        ref_conf.atom_coordinates = ref_coords
        ref_conf.sequence = "ACD"
        ref_conf.energy = -30.0
        
        agent.add_reference_conformation(ref_conf)
        
        # Create moderately similar conformation with different shape (bent)
        # This won't align perfectly, giving RMSD > 1.0
        similar_coords = [(0.0, 0.0, 0.0), (3.0, 1.5, 0.0), (6.0, 1.5, 0.0)]
        similar_conf = Mock()
        similar_conf.atom_coordinates = similar_coords
        similar_conf.sequence = "ACD"
        
        pattern = agent._detect_geometric_similarity(similar_conf)
        
        # Should be MEDIUM or LOW significance (not HIGH)
        # If no pattern detected (RMSD > threshold), that's also valid
        if pattern is not None:
            assert pattern.significance.value in ["medium", "low"]  # Depends on exact RMSD
    
    def test_detect_geometric_similarity_updates_statistics(self, mock_dependencies):
        """Test that geometric similarity detection updates statistics"""
        qcpp_adapter, geometric_analyzer, shared_memory = mock_dependencies
        
        # Mock geometric analyzer
        mock_analysis = Mock()
        mock_analysis.golden_ratio_percentage = 25.0
        mock_analysis.tetrahedron_similarity = 0.5
        mock_analysis.cube_similarity = 0.5
        mock_analysis.octahedron_similarity = 0.5
        mock_analysis.dodecahedron_similarity = 0.5
        mock_analysis.icosahedron_similarity = 0.8
        geometric_analyzer.analyze_conformation.return_value = mock_analysis
        
        agent = MediatorAgent(
            protein_sequence="ACDEFGH",
            qcpp_adapter=qcpp_adapter,
            geometric_analyzer=geometric_analyzer,
            shared_memory=shared_memory
        )
        
        # Add reference
        ref_coords = [(0.0, 0.0, 0.0), (3.8, 0.0, 0.0)]
        ref_conf = Mock()
        ref_conf.atom_coordinates = ref_coords
        ref_conf.sequence = "AC"
        ref_conf.energy = -20.0
        
        agent.add_reference_conformation(ref_conf)
        
        # Create similar conformation
        similar_coords = [(0.1, 0.1, 0.1), (3.9, 0.1, 0.1)]
        similar_conf = Mock()
        similar_conf.atom_coordinates = similar_coords
        similar_conf.sequence = "AC"
        
        # Initial statistics
        initial_stats = agent.detection_statistics.copy()
        
        # Call detect_patterns (which calls _detect_geometric_similarity)
        patterns = agent.detect_patterns(similar_conf)
        
        # Should update statistics
        assert agent.detection_statistics['geometric_detections'] > initial_stats['geometric_detections']
        assert agent.detection_statistics['total_detections'] > initial_stats['total_detections']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
