# Design Document

## Overview

This document describes the architectural design for integrating the Quantum Coherence Protein Predictor (QCPP) and Universal Behavioral Framework (UBF) systems into a unified real-time protein structure prediction platform. The integration creates a bidirectional feedback loop where QCPP provides physics-based guidance during UBF conformational exploration, while UBF provides dynamic validation data to QCPP.

The key innovation is eliminating the separation between exploration (UBF) and validation (QCPP) by making them work together in real-time. This grounds consciousness-based navigation in quantum physics and enables dynamic validation across thousands of conformations.

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    COBATES INTEGRATED                        │
│                                                              │
│  ┌─────────────┐         Real-time          ┌────────────┐ │
│  │    QCPP     │ ←──── Feedback Loop ────→ │    UBF     │ │
│  │             │                             │            │ │
│  │ • QCP calc  │ ──→ Guides moves           │ • Agents   │ │
│  │ • Coherence │ ──→ Evaluates quality      │ • Memory   │ │
│  │ • Phi match │ ──→ Rewards patterns       │ • Moves    │ │
│  │ • Stability │ ←── Gets conformations     │ • Energy   │ │
│  │ • THz spec  │ ←── Dynamic validation     │ • RMSD     │ │
│  └─────────────┘                             └────────────┘ │
│         ↓                                           ↓        │
│    Physics Knowledge                         Exploration     │
│         ↓                                           ↓        │
│         └───────────── Combined Output ────────────┘        │
│                              ↓                               │
│                   Native-like Structure                      │
│                   with Quantum Validation                    │
└─────────────────────────────────────────────────────────────┘
```

### Component Architecture

#### 1. QCPP Integration Layer

**Purpose**: Provides quantum physics calculations to UBF components

**Components**:
- `QCPPIntegrationAdapter`: Wrapper around `QuantumCoherenceProteinPredictor` that provides fast, cached analysis for UBF
- `QCPPMetrics`: Data class containing QCP score, coherence, stability, phi match
- `QCPPCache`: LRU cache for conformation analysis to maintain performance

**Key Methods**:
```python
class QCPPIntegrationAdapter:
    def analyze_conformation(self, conformation: Conformation) -> QCPPMetrics
    def calculate_quantum_alignment(self, conformation: Conformation) -> float
    def get_stability_score(self, conformation: Conformation) -> float
    def get_phi_match_score(self, conformation: Conformation) -> float
```

#### 2. Enhanced Move Evaluator

**Purpose**: Replaces QAAP-based quantum alignment with QCPP-derived metrics

**Components**:
- `IntegratedMoveEvaluator`: Extends `CapabilityBasedMoveEvaluator` with QCPP integration
- Maintains existing 5-factor evaluation system but replaces quantum_alignment calculation

**Evaluation Formula**:
```
weight = (
    physical_feasibility      × 0.2 +  # Unchanged
    quantum_alignment_qcpp    × 0.3 +  # ← QCPP-derived (was QAAP)
    behavioral_preference     × 0.2 +  # Unchanged
    historical_success        × 0.15 + # Unchanged
    goal_alignment           × 0.15    # Unchanged
)

quantum_alignment_qcpp = 0.5 + min(1.0, 
    (qcp_score / 5.0) * 0.4 +      # QCP normalized
    (coherence + 1.0) * 0.3 +       # Coherence shifted
    phi_match * 0.3                 # Phi angle alignment
)
```

#### 3. Physics-Grounded Consciousness

**Purpose**: Maps QCPP metrics to consciousness coordinates

**Components**:
- `PhysicsGroundedConsciousness`: Extends `ConsciousnessState` with QCPP mapping
- Maintains smooth transitions using exponential smoothing

**Mapping Formulas**:
```python
# QCP → Frequency (inverse relationship)
# High QCP (stable) → Low frequency (settled)
# Low QCP (unstable) → High frequency (exploring)
target_frequency = 15.0 - (qcp_score / 0.5)  # Range: 3-15 Hz

# QCPP Coherence → Consciousness Coherence (direct)
target_coherence = 0.2 + (qcpp_coherence * 0.8)  # Range: 0.2-1.0

# Smooth transition (exponential smoothing)
frequency += (target_frequency - frequency) * 0.1
coherence += (target_coherence - coherence) * 0.1
```

#### 4. QCPP-Validated Memory System

**Purpose**: Incorporates QCPP validation into memory significance

**Components**:
- `QCPPValidatedMemory`: Extends `ConformationalMemory` with QCPP metrics
- Enhanced significance calculation incorporating stability scores

**Significance Formula**:
```python
significance = (
    energy_improvement * 0.3 +
    rmsd_improvement * 0.3 +
    qcpp_stability * 0.3 +      # ← New QCPP factor
    novelty * 0.1
)

# High-significance threshold for shared pool
if qcpp_stability > 1.5 and energy_change < -20:
    mark_as_high_significance()
```

#### 5. Dynamic Parameter Adjustment

**Purpose**: Adjusts exploration parameters based on QCPP stability

**Components**:
- `DynamicParameterAdjuster`: Modifies agent parameters in response to QCPP metrics

**Adjustment Rules**:
```python
if qcpp_stability < 1.0:
    # Unstable region → Explore more
    frequency += 2.0  # Increase exploration
    temperature += 50.0
elif qcpp_stability > 2.0:
    # Stable region → Exploit
    frequency -= 1.0  # Decrease exploration
    temperature -= 20.0

# Bounds enforcement
frequency = clamp(frequency, 3.0, 15.0)
temperature = clamp(temperature, 100.0, 500.0)
```

#### 6. Phi Pattern Rewards

**Purpose**: Provides energy bonuses for golden ratio patterns

**Components**:
- `PhiPatternRewardCalculator`: Analyzes phi match and applies bonuses

**Reward Logic**:
```python
if phi_match_score > 0.8:
    # Strong phi patterns → Energy bonus
    energy_bonus = -50.0  # kcal/mol
    move.energy += energy_bonus
```

#### 7. Integrated Trajectory Recorder

**Purpose**: Records comprehensive trajectory data with both UBF and QCPP metrics

**Components**:
- `IntegratedTrajectoryRecorder`: Extends existing trajectory recording
- `TrajectoryAnalyzer`: Computes correlations between QCPP and RMSD

**Data Structure**:
```python
trajectory_point = {
    # UBF metrics
    'iteration': int,
    'rmsd': float,
    'energy': float,
    'consciousness_frequency': float,
    'consciousness_coherence': float,
    
    # QCPP metrics
    'qcp_score': float,
    'field_coherence': float,
    'stability_score': float,
    'phi_match_score': float,
    
    # Timestamp
    'timestamp': int
}
```

## Components and Interfaces

### New Interfaces

#### IQCPPIntegration
```python
class IQCPPIntegration(Protocol):
    """Interface for QCPP integration with UBF."""
    
    def analyze_conformation(self, conformation: Conformation) -> QCPPMetrics:
        """Analyze conformation using QCPP."""
        ...
    
    def calculate_quantum_alignment(self, conformation: Conformation) -> float:
        """Calculate quantum alignment factor for move evaluation."""
        ...
    
    def get_stability_score(self, conformation: Conformation) -> float:
        """Get QCPP stability score."""
        ...
```

#### IPhysicsGroundedConsciousness
```python
class IPhysicsGroundedConsciousness(IConsciousnessState):
    """Extended consciousness interface with QCPP grounding."""
    
    def update_from_qcpp_metrics(self, metrics: QCPPMetrics) -> None:
        """Update consciousness coordinates from QCPP metrics."""
        ...
    
    def get_target_frequency(self, qcp_score: float) -> float:
        """Calculate target frequency from QCP score."""
        ...
    
    def get_target_coherence(self, qcpp_coherence: float) -> float:
        """Calculate target coherence from QCPP coherence."""
        ...
```

### Modified Components

#### MultiAgentCoordinator
**Changes**:
- Add optional `qcpp_integration: Optional[IQCPPIntegration]` parameter
- Pass QCPP integration to agents during initialization
- Record QCPP metrics in trajectory
- Compute QCPP-RMSD correlations after exploration

#### ProteinAgent
**Changes**:
- Accept optional `qcpp_integration: Optional[IQCPPIntegration]` parameter
- Use `PhysicsGroundedConsciousness` when QCPP is enabled
- Pass QCPP integration to move evaluator
- Update consciousness from QCPP metrics after each move

#### CapabilityBasedMoveEvaluator
**Changes**:
- Accept optional `qcpp_integration: Optional[IQCPPIntegration]` parameter
- Replace `_quantum_alignment_factor()` implementation when QCPP enabled
- Apply phi pattern rewards when QCPP enabled

## Data Models

### QCPPMetrics
```python
@dataclass(frozen=True)
class QCPPMetrics:
    """QCPP analysis results for a conformation."""
    qcp_score: float          # Quantum Consciousness Potential
    field_coherence: float    # Field coherence metric
    stability_score: float    # Stability prediction
    phi_match_score: float    # Phi angle matching score
    calculation_time_ms: float  # Performance tracking
```

### QCPPValidatedMemory
```python
@dataclass(frozen=True)
class QCPPValidatedMemory(ConformationalMemory):
    """Memory with QCPP validation metrics."""
    qcpp_metrics: QCPPMetrics
    qcpp_significance: float  # QCPP contribution to significance
```

### IntegratedTrajectoryPoint
```python
@dataclass(frozen=True)
class IntegratedTrajectoryPoint:
    """Trajectory point with both UBF and QCPP metrics."""
    iteration: int
    
    # UBF metrics
    rmsd: float
    energy: float
    consciousness_frequency: float
    consciousness_coherence: float
    
    # QCPP metrics
    qcp_score: float
    field_coherence: float
    stability_score: float
    phi_match_score: float
    
    timestamp: int
```

## Error Handling

### QCPP Calculation Failures
- **Strategy**: Graceful degradation to QAAP-based calculation
- **Implementation**: Try-catch around QCPP calls with fallback
- **Logging**: Warn when falling back to QAAP

### Performance Degradation
- **Strategy**: Adaptive caching and sampling
- **Implementation**: 
  - LRU cache for recent conformations
  - Sample QCPP analysis every N iterations if too slow
  - Configurable QCPP analysis frequency
- **Monitoring**: Track QCPP calculation times

### Invalid Conformations
- **Strategy**: Skip QCPP analysis for invalid structures
- **Implementation**: Validate conformation before QCPP analysis
- **Fallback**: Use previous QCPP metrics or default values

## Testing Strategy

### Unit Tests

#### QCPP Integration Layer
- Test `QCPPIntegrationAdapter` correctly wraps QCPP predictor
- Test `QCPPCache` correctly caches and retrieves results
- Test `QCPPMetrics` data class validation

#### Enhanced Move Evaluator
- Test quantum alignment calculation with QCPP metrics
- Test fallback to QAAP when QCPP unavailable
- Test phi pattern reward application
- Test move weight calculation with QCPP factors

#### Physics-Grounded Consciousness
- Test frequency mapping from QCP scores
- Test coherence mapping from QCPP coherence
- Test smooth transition with exponential smoothing
- Test bounds enforcement (3-15 Hz, 0.2-1.0)

#### QCPP-Validated Memory
- Test significance calculation with QCPP metrics
- Test high-significance threshold detection
- Test memory storage and retrieval with QCPP data

#### Dynamic Parameter Adjustment
- Test parameter increases in unstable regions
- Test parameter decreases in stable regions
- Test bounds enforcement for frequency and temperature

### Integration Tests

#### End-to-End Integration
- Test single agent exploration with QCPP integration
- Test multi-agent exploration with QCPP integration
- Test trajectory recording with both UBF and QCPP metrics
- Test correlation analysis between QCPP and RMSD

#### Backward Compatibility
- Test UBF operates without QCPP when not provided
- Test QCPP operates independently on static PDB files
- Test configuration flag to disable integration

#### Performance Tests
- Test QCPP analysis completes within 5ms per conformation
- Test multi-agent exploration completes within 5 minutes (10 agents, 2000 iterations)
- Test memory overhead remains acceptable (<100MB per agent)
- Test throughput maintains ≥50 conformations/second/agent

### Validation Tests

#### Scientific Validation
- Test correlation between high QCP and low RMSD
- Test correlation between high coherence and low RMSD
- Test phi pattern rewards improve final RMSD
- Test physics-grounded consciousness improves exploration efficiency

## Performance Considerations

### QCPP Calculation Optimization

**Challenge**: QCPP analysis may be computationally expensive

**Solutions**:
1. **Caching**: LRU cache for recently analyzed conformations
2. **Sampling**: Analyze every Nth conformation instead of all
3. **Async Analysis**: Run QCPP analysis in background thread
4. **Incremental Updates**: Only recalculate changed regions

**Target**: <5ms per conformation analysis

### Memory Management

**Challenge**: Storing QCPP metrics for all trajectory points

**Solutions**:
1. **Selective Storage**: Only store QCPP metrics for significant points
2. **Compression**: Use float32 instead of float64 for metrics
3. **Streaming**: Write trajectory to disk incrementally

**Target**: <100MB memory overhead per agent

### Parallelization

**Challenge**: QCPP analysis may bottleneck parallel agents

**Solutions**:
1. **Thread-Safe QCPP**: Ensure QCPP predictor is thread-safe
2. **Per-Agent QCPP**: Each agent gets own QCPP instance
3. **Shared Cache**: Agents share QCPP cache for common conformations

**Target**: Linear scaling up to 10 agents

## Configuration

### Integration Configuration
```python
@dataclass
class QCPPIntegrationConfig:
    """Configuration for QCPP-UBF integration."""
    
    # Enable/disable integration
    enabled: bool = True
    
    # QCPP analysis frequency
    analysis_frequency: int = 1  # Analyze every N iterations
    
    # Caching
    cache_size: int = 1000  # LRU cache size
    
    # Performance
    max_calculation_time_ms: float = 5.0  # Timeout for QCPP
    
    # Phi pattern rewards
    phi_reward_threshold: float = 0.8  # Min phi match for reward
    phi_reward_energy: float = -50.0  # Energy bonus (kcal/mol)
    
    # Dynamic parameter adjustment
    enable_dynamic_adjustment: bool = True
    stability_low_threshold: float = 1.0
    stability_high_threshold: float = 2.0
    
    # Consciousness grounding
    enable_physics_grounding: bool = True
    smoothing_factor: float = 0.1  # Exponential smoothing
```

### Backward Compatibility
```python
# Without QCPP (existing behavior)
coordinator = MultiAgentCoordinator(
    protein_sequence="ACDEFGH"
)

# With QCPP integration (new behavior)
qcpp = QCPPIntegrationAdapter(
    predictor=QuantumCoherenceProteinPredictor(),
    config=QCPPIntegrationConfig()
)
coordinator = MultiAgentCoordinator(
    protein_sequence="ACDEFGH",
    qcpp_integration=qcpp
)
```

## Migration Path

### Phase 1: Core Integration
1. Implement `QCPPIntegrationAdapter` and `QCPPMetrics`
2. Modify `CapabilityBasedMoveEvaluator` to accept QCPP integration
3. Add QCPP-based quantum alignment calculation
4. Add unit tests for integration layer

### Phase 2: Consciousness Grounding
1. Implement `PhysicsGroundedConsciousness`
2. Add QCPP metric mapping to consciousness coordinates
3. Modify `ProteinAgent` to use physics-grounded consciousness
4. Add unit tests for consciousness grounding

### Phase 3: Memory and Rewards
1. Implement `QCPPValidatedMemory`
2. Add QCPP metrics to significance calculation
3. Implement phi pattern reward calculator
4. Add unit tests for memory and rewards

### Phase 4: Trajectory and Analysis
1. Implement `IntegratedTrajectoryRecorder`
2. Add QCPP metrics to trajectory points
3. Implement correlation analysis
4. Add integration tests for end-to-end flow

### Phase 5: Optimization
1. Implement caching layer
2. Add performance monitoring
3. Optimize QCPP calculation frequency
4. Add performance tests

## Future Enhancements

### THz Spectrum Integration
- Use QCPP's THz spectrum predictions for validation
- Compare predicted vs experimental THz signatures
- Use THz matching as additional move evaluation factor

### Multi-Scale Integration
- Use QCPP for coarse-grained guidance
- Use UBF for fine-grained exploration
- Hierarchical exploration strategy

### Adaptive QCPP Frequency
- Analyze more frequently in promising regions
- Reduce frequency in well-explored regions
- Machine learning to predict when QCPP analysis is valuable

### Collective QCPP Learning
- Agents share QCPP-validated conformational patterns
- Build library of quantum-coherent structural motifs
- Transfer learning across different proteins
