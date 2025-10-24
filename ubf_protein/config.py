# config.py - System configuration

# Consciousness parameters (global bounds)
FREQUENCY_MIN = 3.0
FREQUENCY_MAX = 15.0
COHERENCE_MIN = 0.2
COHERENCE_MAX = 1.0
BEHAVIORAL_STATE_REGEN_THRESHOLD = 0.3

# Memory parameters (base values, can be overridden by AdaptiveConfig)
MEMORY_SIGNIFICANCE_THRESHOLD = 0.3
MAX_MEMORIES_PER_AGENT = 50
MEMORY_INFLUENCE_MIN = 0.8
MEMORY_INFLUENCE_MAX = 1.5
SHARED_MEMORY_SIGNIFICANCE_THRESHOLD = 0.7
MAX_SHARED_MEMORY_POOL_SIZE = 10000

# Memory significance calculation (simplified 3-factor approach)
SIGNIFICANCE_ENERGY_CHANGE_WEIGHT = 0.5
SIGNIFICANCE_STRUCTURAL_NOVELTY_WEIGHT = 0.3
SIGNIFICANCE_RMSD_IMPROVEMENT_WEIGHT = 0.2

# Local minima detection (base values, scaled by AdaptiveConfig)
BASE_STUCK_DETECTION_WINDOW = 20
BASE_STUCK_DETECTION_THRESHOLD = 10.0  # kJ/mol
ESCAPE_FREQUENCY_BOOST_MODERATE = 1.0
ESCAPE_FREQUENCY_BOOST_LARGE = 2.0
ESCAPE_FREQUENCY_BOOST_MAXIMUM = 3.0
ESCAPE_COHERENCE_REDUCTION = 0.1

# Performance targets
TARGET_DECISION_LATENCY_MS = 2.0
TARGET_MEMORY_RETRIEVAL_US = 10.0
TARGET_AGENT_MEMORY_MB = 50.0
TARGET_PYPY_SPEEDUP = 2.0

# Multi-agent diversity
AGENT_PROFILE_CAUTIOUS_RATIO = 0.33
AGENT_PROFILE_BALANCED_RATIO = 0.34
AGENT_PROFILE_AGGRESSIVE_RATIO = 0.33

# Physics integration (composite quantum alignment factor)
QAAP_CONTRIBUTION_MIN = 0.7
QAAP_CONTRIBUTION_MAX = 1.3
RESONANCE_CONTRIBUTION_MIN = 0.9
RESONANCE_CONTRIBUTION_MAX = 1.2
WATER_SHIELDING_CONTRIBUTION_MIN = 0.95
WATER_SHIELDING_CONTRIBUTION_MAX = 1.05
GAMMA_FREQUENCY_HZ = 40.0
COHERENCE_TIME_FS = 408.0
WATER_SHIELDING_FACTOR = 3.57  # nm⁻¹

# Composite factor ranges (simplified 5-factor evaluation)
PHYSICAL_FEASIBILITY_RANGE = (0.1, 2.0)
QUANTUM_ALIGNMENT_RANGE = (0.5, 1.5)
BEHAVIORAL_PREFERENCE_RANGE = (0.5, 2.5)
HISTORICAL_SUCCESS_RANGE = (0.8, 1.8)
GOAL_ALIGNMENT_MAX_BOOST = 10.0

# Adaptive configuration
PROTEIN_SIZE_SMALL_THRESHOLD = 50  # residues
PROTEIN_SIZE_LARGE_THRESHOLD = 150  # residues
THRESHOLD_SCALING_BASELINE = 50  # residues for 1.0x scaling

# Visualization and monitoring
VISUALIZATION_STREAM_INTERVAL = 10  # iterations
TRAJECTORY_BUFFER_MAX_SIZE = 1000  # snapshots
ENERGY_LANDSCAPE_PROJECTION_METHOD = 'PCA'  # or 't-SNE'

# Checkpoint and resume
CHECKPOINT_AUTO_SAVE_INTERVAL = 100  # iterations
CHECKPOINT_ROTATION_KEEP_COUNT = 5  # keep last N checkpoints
CHECKPOINT_FORMAT_VERSION = '1.0'

# Validation targets
TARGET_RMSD_ANGSTROM = 3.0
TARGET_GDT_TS = 70.0
TARGET_LEARNING_IMPROVEMENT_PERCENT = 50.0

# Agent diversity profiles
AGENT_DIVERSITY_PROFILES = {
    'cautious': {
        'profile_name': 'cautious',
        'frequency_range': (4.0, 7.0),
        'coherence_range': (0.7, 1.0),
        'description': 'Low energy, high focus - careful local exploration'
    },
    'balanced': {
        'profile_name': 'balanced',
        'frequency_range': (7.0, 10.0),
        'coherence_range': (0.5, 0.8),
        'description': 'Moderate energy and focus - balanced exploration'
    },
    'aggressive': {
        'profile_name': 'aggressive',
        'frequency_range': (10.0, 15.0),
        'coherence_range': (0.3, 0.6),
        'description': 'High energy, lower focus - bold exploration, escapes minima'
    }
}

# Consciousness update rules (from UBF)
CONSCIOUSNESS_UPDATE_RULES = {
    'energy_decrease_large': {'frequency': +0.5, 'coherence': +0.1},
    'energy_decrease_small': {'frequency': +0.2, 'coherence': +0.05},
    'energy_increase': {'frequency': -0.3, 'coherence': -0.05},
    'structure_collapse': {'frequency': -1.0, 'coherence': -0.2},
    'stable_minimum_found': {'frequency': +0.3, 'coherence': +0.15},
    'helix_formation': {'frequency': +0.2, 'coherence': +0.08},
    'sheet_formation': {'frequency': +0.2, 'coherence': +0.08},
    'hydrophobic_core_formed': {'frequency': +0.4, 'coherence': +0.1},
    'stuck_in_local_minimum': {'frequency': +1.0, 'coherence': -0.1},  # Boost to escape!
}