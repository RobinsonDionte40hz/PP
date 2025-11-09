# Requirements Document

## Introduction

This specification defines two new modules for the QCPP-UBF protein structure prediction platform:

1. **Geometric Attractor Module (Updated Version)**: A production-ready module that analyzes spatial relationships within protein conformations and outputs percentage scores representing the strength of detected geometric patterns. This module extends the existing experimental geometric attractor analysis to provide real-time, actionable metrics during protein folding simulations.

2. **Mediator Agents Module**: A new class of specialized agents that act as information relays between the QCPP system and exploration agents. Mediators detect THz resonance patterns, folding dynamics, and geometric similarities, while supporting memory flow and data caching to improve coordination and system efficiency.

Both modules integrate seamlessly with the existing `test_protein.py` workflow and the UBF multi-agent system.

## Glossary

- **QCPP System**: Quantum Coherence Protein Predictor - physics-based stability prediction system
- **UBF System**: Universal Behavioral Framework - consciousness-based conformational exploration system
- **Geometric Attractor**: A spatial configuration in conformational space that exhibits golden ratio (φ) patterns or Platonic solid symmetries
- **Golden Ratio (φ)**: Mathematical constant ≈ 1.618, appears in dodecahedron and icosahedron geometries
- **Platonic Solid**: One of five regular polyhedra (tetrahedron, cube, octahedron, dodecahedron, icosahedron)
- **THz Signature**: Terahertz frequency spectrum representing vibrational modes of a protein conformation
- **Mediator Agent**: Specialized agent that detects patterns and relays information between QCPP and exploration agents
- **Conformation**: A specific 3D spatial arrangement of a protein's atoms
- **RMSD**: Root Mean Square Deviation - metric for structural similarity (Ångströms)
- **Memory System**: UBF component storing significant conformational transitions
- **Shared Memory Pool**: Collective memory accessible to all agents for collaborative learning
- **Consciousness Coordinates**: 2D state space (frequency 3-15 Hz, coherence 0.2-1.0) guiding agent behavior
- **Move Evaluation**: Process of scoring potential conformational changes using multiple factors

## Requirements

### Requirement 1: Geometric Attractor Module - Core Analysis

**User Story:** As a protein researcher, I want to analyze protein conformations for geometric patterns in real-time, so that I can understand which spatial relationships contribute to successful folding.

#### Acceptance Criteria

1. WHEN a protein conformation is provided to the Geometric Attractor Module, THE Module SHALL calculate percentage scores for golden ratio (φ) patterns in distance ratios
2. WHEN analyzing spatial relationships, THE Module SHALL compute percentage scores for each of the five Platonic solid similarities (tetrahedron, cube, octahedron, dodecahedron, icosahedron)
3. WHEN calculating symmetry metrics, THE Module SHALL output percentage scores for rotational symmetry and local symmetry
4. WHEN processing a conformation, THE Module SHALL complete analysis within 50 milliseconds for proteins up to 200 residues
5. WHERE the conformation has fewer than 4 residues, THE Module SHALL return zero scores with a warning message

### Requirement 2: Geometric Attractor Module - Integration with test_protein.py

**User Story:** As a developer, I want the Geometric Attractor Module to integrate seamlessly with test_protein.py, so that users can access geometric analysis without modifying their existing workflow.

#### Acceptance Criteria

1. WHEN test_protein.py executes a protein test, THE Geometric Attractor Module SHALL be automatically invoked if a PDB file is available
2. WHEN the module completes analysis, THE Module SHALL output results in a structured dictionary format compatible with JSON serialization
3. WHEN displaying results, THE Module SHALL print percentage scores with one decimal place precision
4. WHEN saving results, THE Module SHALL include geometric analysis data in the test results JSON file
5. IF the module encounters an error, THEN THE Module SHALL log a warning and allow test_protein.py to continue execution

### Requirement 3: Geometric Attractor Module - Performance and Caching

**User Story:** As a system administrator, I want the Geometric Attractor Module to use efficient caching, so that repeated analyses of similar conformations do not degrade performance.

#### Acceptance Criteria

1. WHEN analyzing a conformation, THE Module SHALL generate a hash based on CA atom coordinates
2. WHEN a conformation hash matches a cached entry, THE Module SHALL return cached results within 1 millisecond
3. WHEN the cache exceeds 5000 entries, THE Module SHALL evict least-recently-used entries
4. WHEN calculating distance ratios, THE Module SHALL sample intelligently to maintain O(n²) complexity rather than O(n⁴)
5. WHEN memory usage exceeds 100 MB, THE Module SHALL clear the cache and log a warning

### Requirement 4: Mediator Agents - Agent Architecture

**User Story:** As a system architect, I want Mediator Agents to be a distinct agent class, so that they can perform specialized pattern detection without interfering with exploration agents.

#### Acceptance Criteria

1. THE Mediator Agent class SHALL implement the IProteinAgent interface from ubf_protein/interfaces.py
2. WHEN initialized, THE Mediator Agent SHALL accept configuration parameters for detection thresholds and relay frequency
3. WHEN operating, THE Mediator Agent SHALL maintain its own consciousness coordinates separate from exploration agents
4. WHEN detecting patterns, THE Mediator Agent SHALL not modify protein conformations
5. THE Mediator Agent SHALL expose methods for pattern detection, information relay, and cache management

### Requirement 5: Mediator Agents - THz Resonance Detection

**User Story:** As a quantum physicist, I want Mediator Agents to detect THz resonance patterns, so that I can identify conformations with similar vibrational signatures.

#### Acceptance Criteria

1. WHEN analyzing a conformation, THE Mediator Agent SHALL calculate THz signature using QCPP integration
2. WHEN comparing THz signatures, THE Mediator Agent SHALL compute similarity scores using spectral correlation
3. WHEN a THz signature similarity exceeds 0.7 threshold, THE Mediator Agent SHALL flag the conformation as resonant
4. WHEN multiple conformations share resonant signatures, THE Mediator Agent SHALL cluster them and report cluster statistics
5. WHEN THz analysis completes, THE Mediator Agent SHALL cache results indexed by conformation hash

### Requirement 6: Mediator Agents - Folding Dynamics Detection

**User Story:** As a structural biologist, I want Mediator Agents to detect secondary structure formation, so that I can track helix, sheet, and turn formation during exploration.

#### Acceptance Criteria

1. WHEN analyzing a conformation, THE Mediator Agent SHALL identify alpha-helix regions using phi-psi angle criteria
2. WHEN detecting beta-sheets, THE Mediator Agent SHALL identify parallel and antiparallel strand arrangements
3. WHEN finding turns, THE Mediator Agent SHALL classify turn types (I, II, III, IV) based on geometry
4. WHEN secondary structure changes occur, THE Mediator Agent SHALL calculate percentage scores for each structure type
5. WHEN reporting dynamics, THE Mediator Agent SHALL output percentage of residues in helix, sheet, turn, and coil states

### Requirement 7: Mediator Agents - Geometric Similarity Detection

**User Story:** As a protein engineer, I want Mediator Agents to detect geometric similarities between conformations, so that I can identify convergent folding pathways.

#### Acceptance Criteria

1. WHEN comparing two conformations, THE Mediator Agent SHALL calculate RMSD between CA atom positions
2. WHEN RMSD is below 2.0 Ångströms, THE Mediator Agent SHALL flag conformations as geometrically similar
3. WHEN detecting similar conformations, THE Mediator Agent SHALL invoke the Geometric Attractor Module for detailed analysis
4. WHEN multiple conformations cluster geometrically, THE Mediator Agent SHALL identify the centroid conformation
5. WHEN reporting similarities, THE Mediator Agent SHALL output percentage scores for structural overlap

### Requirement 8: Mediator Agents - Information Relay to QCPP

**User Story:** As a system integrator, I want Mediator Agents to relay information to the QCPP system, so that physics-based analysis can inform agent exploration.

#### Acceptance Criteria

1. WHEN a Mediator Agent detects a significant pattern, THE Agent SHALL invoke QCPP analysis via QCPPIntegrationAdapter
2. WHEN QCPP analysis completes, THE Mediator Agent SHALL broadcast results to all exploration agents via shared memory
3. WHEN relaying information, THE Mediator Agent SHALL include pattern type, significance score, and QCPP metrics
4. WHEN broadcast frequency exceeds 10 messages per second, THE Mediator Agent SHALL throttle to prevent memory overflow
5. IF QCPP analysis fails, THEN THE Mediator Agent SHALL log the error and continue operation

### Requirement 9: Mediator Agents - Information Relay to Exploration Agents

**User Story:** As an exploration agent, I want to receive pattern information from Mediator Agents, so that I can adjust my search strategy based on detected attractors.

#### Acceptance Criteria

1. WHEN a Mediator Agent broadcasts a pattern, THE Exploration Agent SHALL receive the message via shared memory pool
2. WHEN receiving geometric attractor information, THE Exploration Agent SHALL increase move evaluation scores toward similar geometries
3. WHEN receiving THz resonance information, THE Exploration Agent SHALL prioritize moves that maintain resonant frequencies
4. WHEN receiving folding dynamics information, THE Exploration Agent SHALL adjust secondary structure formation preferences
5. WHEN pattern information is older than 100 iterations, THE Exploration Agent SHALL reduce its influence weight by 50%

### Requirement 10: Mediator Agents - Memory Flow and Caching

**User Story:** As a performance engineer, I want Mediator Agents to implement efficient caching, so that pattern detection does not become a performance bottleneck.

#### Acceptance Criteria

1. WHEN analyzing a conformation, THE Mediator Agent SHALL check cache before performing expensive calculations
2. WHEN cache hit occurs, THE Mediator Agent SHALL return cached results within 5 milliseconds
3. WHEN storing cache entries, THE Mediator Agent SHALL use conformation hash as key and include timestamp
4. WHEN cache size exceeds 10,000 entries, THE Mediator Agent SHALL evict entries older than 1 hour
5. WHEN memory pressure is detected, THE Mediator Agent SHALL reduce cache size by 50%

### Requirement 11: Mediator Agents - Coordination with Multi-Agent System

**User Story:** As a multi-agent coordinator, I want Mediator Agents to integrate with the existing MultiAgentCoordinator, so that they can operate alongside exploration agents.

#### Acceptance Criteria

1. WHEN MultiAgentCoordinator initializes, THE Coordinator SHALL create Mediator Agents based on configuration
2. WHEN running parallel exploration, THE Coordinator SHALL execute Mediator Agent detection cycles every 20 iterations
3. WHEN Mediator Agents detect patterns, THE Coordinator SHALL aggregate statistics across all Mediators
4. WHEN exploration completes, THE Coordinator SHALL include Mediator Agent statistics in final results
5. WHERE Mediator Agents are disabled in configuration, THE Coordinator SHALL operate without them

### Requirement 12: Geometric Attractor Module - Output Format

**User Story:** As a data analyst, I want geometric analysis results in a standardized format, so that I can perform statistical analysis across multiple protein tests.

#### Acceptance Criteria

1. THE Geometric Attractor Module SHALL output results as a Python dictionary with nested structure
2. WHEN outputting golden ratio analysis, THE Module SHALL include keys: 'percentage', 'total_patterns', 'total_ratios_analyzed'
3. WHEN outputting symmetry analysis, THE Module SHALL include keys: 'rotational', 'local', 'radius_of_gyration', 'asphericity'
4. WHEN outputting Platonic similarities, THE Module SHALL include keys for all five solids with scores 0.0-1.0
5. WHEN outputting QCPP components, THE Module SHALL include keys: 'golden_correlation', 'doubling_correlation'

### Requirement 13: Mediator Agents - Configuration and Tuning

**User Story:** As a system operator, I want to configure Mediator Agent behavior, so that I can tune detection sensitivity for different protein types.

#### Acceptance Criteria

1. THE Mediator Agent configuration SHALL include parameters for THz similarity threshold (default: 0.7)
2. THE Mediator Agent configuration SHALL include parameters for geometric similarity threshold (default: 2.0 Å RMSD)
3. THE Mediator Agent configuration SHALL include parameters for relay frequency (default: every 20 iterations)
4. THE Mediator Agent configuration SHALL include parameters for cache size (default: 10,000 entries)
5. WHEN configuration is invalid, THE Mediator Agent SHALL raise a ValueError with descriptive message

### Requirement 14: Integration Testing and Validation

**User Story:** As a quality assurance engineer, I want comprehensive tests for both modules, so that I can verify correct behavior across different scenarios.

#### Acceptance Criteria

1. THE test suite SHALL include unit tests for Geometric Attractor Module covering all analysis functions
2. THE test suite SHALL include unit tests for Mediator Agent covering pattern detection and relay functions
3. THE test suite SHALL include integration tests verifying Mediator Agent coordination with MultiAgentCoordinator
4. THE test suite SHALL include performance tests verifying analysis completes within time budgets
5. THE test suite SHALL achieve at least 85% code coverage for both modules

### Requirement 15: Documentation and Examples

**User Story:** As a new user, I want clear documentation and examples, so that I can understand how to use the new modules effectively.

#### Acceptance Criteria

1. THE Geometric Attractor Module SHALL include docstrings for all public methods following Google style
2. THE Mediator Agent class SHALL include docstrings for all public methods following Google style
3. THE documentation SHALL include a usage example showing integration with test_protein.py
4. THE documentation SHALL include a usage example showing Mediator Agent configuration
5. THE documentation SHALL include performance characteristics and expected output formats
