# Design Document

## Overview

This design document describes the architecture and implementation strategy for two new modules in the QCPP-UBF protein structure prediction platform:

1. **Geometric Attractor Module**: A production-ready analysis module that computes percentage scores for geometric patterns in protein conformations
2. **Mediator Agents Module**: A specialized agent system that detects patterns and relays information between QCPP and exploration agents

Both modules are designed to integrate seamlessly with the existing UBF architecture while maintaining the system's performance characteristics and SOLID design principles.

### Design Goals

- **Performance**: Geometric analysis < 50ms, cache hits < 1ms, Mediator operations < 10ms
- **Modularity**: Clean interfaces following existing UBF patterns
- **Compatibility**: Zero breaking changes to existing test_protein.py and UBF components
- **Scalability**: Support for 100+ agents and 10,000+ cached conformations
- **Maintainability**: Comprehensive documentation and test coverage > 85%

## Architecture

### System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                    QCPP-UBF Integration Layer                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐         ┌──────────────────┐             │
│  │  QCPP System     │◄────────┤ Mediator Agents  │             │
│  │  (Physics)       │         │  (Pattern Relay) │             │
│  └──────────────────┘         └────────┬─────────┘             │
│           │                             │                        │
│           │                             ▼                        │
│           │                    ┌─────────────────┐              │
│           │                    │ Shared Memory   │              │
│           │                    │ Pool            │              │
│           │                    └────────┬────────┘              │
│           │                             │                        │
│           ▼                             ▼                        │
│  ┌──────────────────┐         ┌─────────────────┐              │
│  │ Geometric        │◄────────┤ Exploration     │              │
│  │ Attractor Module │         │ Agents          │              │
│  └──────────────────┘         └─────────────────┘              │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Module Relationships

- **Geometric Attractor Module**: Standalone analyzer invoked by test_protein.py and Mediator Agents
- **Mediator Agents**: Specialized agents that coordinate between QCPP and exploration agents
- **Shared Memory Pool**: Communication channel for pattern information
- **QCPP Integration**: Existing adapter used by Mediators for physics analysis



## Components and Interfaces

### 1. Geometric Attractor Module

#### 1.1 GeometricAttractorAnalyzer Class

**Location**: `ubf_protein/geometric_attractor.py`

**Purpose**: Main analyzer class providing geometric pattern detection with caching

**Public Interface**:
```python
class GeometricAttractorAnalyzer:
    def __init__(self, cache_size: int = 5000, phi_tolerance: float = 0.1):
        """Initialize analyzer with configurable cache and tolerance"""
        
    def analyze_conformation(self, 
                           conformation: Conformation,
                           pdb_file: Optional[Path] = None) -> GeometricAnalysisResult:
        """Analyze conformation for geometric patterns (cached)"""
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics for monitoring"""
        
    def clear_cache(self) -> None:
        """Clear analysis cache"""
```

**Key Methods**:
- `_calculate_golden_ratio_patterns()`: Compute φ patterns in distance ratios
- `_calculate_platonic_similarities()`: Compute similarity to 5 Platonic solids
- `_calculate_symmetry_metrics()`: Compute rotational and local symmetry
- `_generate_conformation_hash()`: Create cache key from CA coordinates
- `_sample_distance_ratios()`: Intelligent O(n²) sampling strategy

#### 1.2 GeometricAnalysisResult Data Class

**Purpose**: Immutable result container for geometric analysis

**Structure**:
```python
@dataclass(frozen=True)
class GeometricAnalysisResult:
    # Golden ratio analysis
    golden_ratio_percentage: float
    golden_ratio_count: int
    total_ratios_analyzed: int
    
    # Symmetry metrics
    rotational_symmetry: float  # 0.0-1.0
    local_symmetry: float       # 0.0-1.0
    radius_of_gyration: float   # Ångströms
    asphericity: float          # 0.0-1.0
    
    # Platonic solid similarities (0.0-1.0)
    tetrahedron_similarity: float
    cube_similarity: float
    octahedron_similarity: float
    dodecahedron_similarity: float
    icosahedron_similarity: float
    
    # QCPP component correlations
    golden_correlation: float
    doubling_correlation: float
    
    # Performance metrics
    calculation_time_ms: float
    cache_hit: bool
```



### 2. Mediator Agents Module

#### 2.1 MediatorAgent Class

**Location**: `ubf_protein/mediator_agent.py`

**Purpose**: Specialized agent for pattern detection and information relay

**Public Interface**:
```python
class MediatorAgent(IProteinAgent):
    def __init__(self,
                 protein_sequence: str,
                 qcpp_adapter: QCPPIntegrationAdapter,
                 geometric_analyzer: GeometricAttractorAnalyzer,
                 shared_memory: ISharedMemoryPool,
                 config: MediatorConfig):
        """Initialize Mediator Agent with dependencies"""
        
    def detect_patterns(self, conformation: Conformation) -> List[PatternDetection]:
        """Detect THz, folding, and geometric patterns"""
        
    def relay_to_qcpp(self, pattern: PatternDetection) -> QCPPMetrics:
        """Request QCPP analysis for detected pattern"""
        
    def broadcast_to_agents(self, pattern: PatternDetection, qcpp_metrics: QCPPMetrics) -> None:
        """Broadcast pattern information via shared memory"""
        
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Return pattern detection statistics"""
        
    def explore_step(self) -> ConformationalOutcome:
        """Required by IProteinAgent - performs detection cycle"""
```

**Key Methods**:
- `_detect_thz_resonance()`: Analyze THz signatures and cluster similar conformations
- `_detect_folding_dynamics()`: Identify secondary structure formation
- `_detect_geometric_similarity()`: Find geometrically similar conformations
- `_check_cache()`: Query pattern cache before expensive calculations
- `_throttle_broadcast()`: Prevent memory overflow from excessive messaging

#### 2.2 MediatorConfig Data Class

**Purpose**: Configuration for Mediator Agent behavior

**Structure**:
```python
@dataclass
class MediatorConfig:
    # Detection thresholds
    thz_similarity_threshold: float = 0.7
    geometric_similarity_threshold: float = 2.0  # Ångströms RMSD
    secondary_structure_min_length: int = 4      # Minimum helix/sheet length
    
    # Relay configuration
    relay_frequency: int = 20           # Detect patterns every N iterations
    broadcast_throttle_rate: int = 10   # Max broadcasts per second
    
    # Cache configuration
    cache_size: int = 10000
    cache_ttl_seconds: int = 3600       # 1 hour
    
    # Performance tuning
    enable_thz_detection: bool = True
    enable_folding_detection: bool = True
    enable_geometric_detection: bool = True
```



#### 2.3 PatternDetection Data Class

**Purpose**: Container for detected pattern information

**Structure**:
```python
@dataclass
class PatternDetection:
    pattern_type: PatternType  # Enum: THz, Folding, Geometric
    detection_time: int        # Timestamp in milliseconds
    conformation_hash: str     # Hash of analyzed conformation
    significance: float        # 0.0-1.0 score
    
    # Pattern-specific data
    thz_data: Optional[THzResonanceData] = None
    folding_data: Optional[FoldingDynamicsData] = None
    geometric_data: Optional[GeometricSimilarityData] = None
    
    # QCPP validation
    qcpp_metrics: Optional[QCPPMetrics] = None
    qcpp_validated: bool = False

@dataclass
class THzResonanceData:
    frequencies: np.ndarray
    intensities: np.ndarray
    cluster_id: int
    cluster_size: int
    similarity_score: float

@dataclass
class FoldingDynamicsData:
    helix_percentage: float
    sheet_percentage: float
    turn_percentage: float
    coil_percentage: float
    helix_regions: List[Tuple[int, int]]  # (start, end) residue indices
    sheet_regions: List[Tuple[int, int]]

@dataclass
class GeometricSimilarityData:
    reference_hash: str
    rmsd: float
    geometric_analysis: GeometricAnalysisResult
    structural_overlap: float  # Percentage
```

#### 2.4 PatternType Enum

**Purpose**: Classification of detected patterns

```python
class PatternType(Enum):
    THZ_RESONANCE = "thz_resonance"
    FOLDING_DYNAMICS = "folding_dynamics"
    GEOMETRIC_SIMILARITY = "geometric_similarity"
```



### 3. Integration Components

#### 3.1 MultiAgentCoordinator Extension

**Location**: `ubf_protein/multi_agent_coordinator.py` (modifications)

**New Parameters**:
```python
class MultiAgentCoordinator:
    def __init__(self,
                 # ... existing parameters ...
                 enable_mediators: bool = False,
                 mediator_count: int = 2,
                 mediator_config: Optional[MediatorConfig] = None):
```

**New Methods**:
```python
def initialize_mediators(self) -> None:
    """Create and initialize Mediator Agents"""
    
def run_mediator_cycle(self, iteration: int) -> List[PatternDetection]:
    """Execute pattern detection cycle for all Mediators"""
    
def get_mediator_statistics(self) -> Dict[str, Any]:
    """Aggregate statistics from all Mediators"""
```

#### 3.2 test_protein.py Integration

**Location**: `test_protein.py` (modifications)

**New Function**:
```python
def analyze_geometric_attractors_v2(pdb_file: Path, 
                                   sequence: str,
                                   best_conformation: Conformation,
                                   qcpp_adapter: QCPPIntegrationAdapter) -> Optional[dict]:
    """
    Analyze protein structure using updated Geometric Attractor Module.
    
    This replaces the existing analyze_geometric_attractors() function
    with the new production-ready module.
    """
```

**Configuration Addition**:
```python
parser.add_argument('--enable-mediators', 
                    action='store_true',
                    help='Enable Mediator Agents for pattern detection')
parser.add_argument('--mediator-count',
                    type=int,
                    default=2,
                    help='Number of Mediator Agents (default: 2)')
```



## Data Models

### Conformation Hash Generation

**Purpose**: Create unique, deterministic identifiers for conformations to enable caching

**Algorithm**:
```python
def generate_conformation_hash(conformation: Conformation) -> str:
    """
    Generate SHA256 hash from CA atom coordinates.
    
    Steps:
    1. Extract CA coordinates as numpy array
    2. Round to 2 decimal places (0.01 Å precision)
    3. Flatten to 1D array
    4. Convert to bytes
    5. Compute SHA256 hash
    6. Return first 16 characters (sufficient uniqueness)
    """
    ca_coords = np.array(conformation.atom_coordinates)
    rounded = np.round(ca_coords, decimals=2)
    flattened = rounded.flatten()
    hash_obj = hashlib.sha256(flattened.tobytes())
    return hash_obj.hexdigest()[:16]
```

### Cache Entry Structure

**Purpose**: Store analysis results with metadata for eviction policy

```python
@dataclass
class CacheEntry:
    key: str                    # Conformation hash
    result: Any                 # Analysis result (GeometricAnalysisResult or PatternDetection)
    timestamp: int              # Creation time (milliseconds)
    access_count: int           # Number of cache hits
    last_access: int            # Last access time (milliseconds)
    size_bytes: int             # Memory footprint estimate
```

### LRU Cache Implementation

**Purpose**: Efficient cache with automatic eviction

**Strategy**:
- Use OrderedDict for O(1) access and LRU tracking
- Evict least-recently-used entries when size limit reached
- Track access patterns for monitoring
- Support TTL-based expiration



## Error Handling

### Geometric Attractor Module

**Error Scenarios**:

1. **Invalid Conformation**: Fewer than 4 residues
   - Action: Return zero scores with warning flag
   - Log: Warning level
   - Impact: Non-critical, allow execution to continue

2. **PDB Parsing Failure**: Cannot read PDB file
   - Action: Return empty analysis with error flag
   - Log: Warning level
   - Impact: Non-critical, skip geometric analysis

3. **Memory Pressure**: Cache exceeds memory limit
   - Action: Clear cache, log warning, continue
   - Log: Warning level
   - Impact: Performance degradation (cache misses)

4. **Calculation Timeout**: Analysis exceeds 100ms
   - Action: Return partial results, set timeout flag
   - Log: Warning level
   - Impact: Incomplete analysis, but system continues

### Mediator Agents

**Error Scenarios**:

1. **QCPP Analysis Failure**: QCPP adapter returns error
   - Action: Skip QCPP validation, use pattern detection only
   - Log: Warning level
   - Impact: Reduced accuracy, but pattern detection continues

2. **Broadcast Overflow**: Shared memory pool full
   - Action: Throttle broadcasts, drop low-significance patterns
   - Log: Warning level
   - Impact: Reduced information flow, but system stable

3. **THz Calculation Failure**: Cannot compute THz signature
   - Action: Skip THz detection for this conformation
   - Log: Info level
   - Impact: Missing THz data, other detections continue

4. **Cache Corruption**: Invalid cache entry detected
   - Action: Clear corrupted entry, log error, continue
   - Log: Error level
   - Impact: Single cache miss, system recovers

**Error Handling Principles**:
- Fail gracefully: Never crash the main exploration loop
- Log appropriately: Errors for critical issues, warnings for degraded performance
- Provide fallbacks: Default values or partial results when possible
- Monitor health: Track error rates for system health monitoring



## Testing Strategy

### Unit Tests

#### Geometric Attractor Module Tests (`test_geometric_attractor.py`)

**Test Coverage**:
1. `test_golden_ratio_calculation`: Verify φ pattern detection accuracy
2. `test_platonic_similarity_calculation`: Verify similarity scores for known geometries
3. `test_symmetry_metrics`: Verify rotational and local symmetry calculations
4. `test_cache_hit_performance`: Verify cache returns results < 1ms
5. `test_cache_eviction`: Verify LRU eviction when size limit reached
6. `test_invalid_conformation`: Verify graceful handling of < 4 residues
7. `test_pdb_parsing_failure`: Verify error handling for invalid PDB files
8. `test_conformation_hash_determinism`: Verify same conformation produces same hash
9. `test_analysis_performance`: Verify analysis completes < 50ms for 200 residues
10. `test_output_format`: Verify result structure matches specification

**Test Proteins**:
- Small (< 50 residues): 1VII (35 residues)
- Medium (50-150 residues): 1UBQ (76 residues)
- Large (> 150 residues): 1LYZ (129 residues)

#### Mediator Agent Tests (`test_mediator_agent.py`)

**Test Coverage**:
1. `test_thz_resonance_detection`: Verify THz signature clustering
2. `test_folding_dynamics_detection`: Verify secondary structure identification
3. `test_geometric_similarity_detection`: Verify RMSD-based similarity
4. `test_qcpp_relay`: Verify QCPP adapter invocation
5. `test_broadcast_to_agents`: Verify shared memory communication
6. `test_cache_performance`: Verify pattern cache < 5ms retrieval
7. `test_throttle_mechanism`: Verify broadcast rate limiting
8. `test_configuration_validation`: Verify invalid config raises ValueError
9. `test_pattern_significance_scoring`: Verify significance calculation
10. `test_mediator_statistics`: Verify statistics aggregation

### Integration Tests

#### MultiAgentCoordinator Integration (`test_mediator_integration.py`)

**Test Coverage**:
1. `test_mediator_initialization`: Verify Mediators created with coordinator
2. `test_mediator_cycle_execution`: Verify detection cycles run every 20 iterations
3. `test_pattern_broadcast_to_explorers`: Verify exploration agents receive patterns
4. `test_mediator_statistics_aggregation`: Verify coordinator aggregates Mediator stats
5. `test_disabled_mediators`: Verify system works without Mediators enabled
6. `test_multiple_mediators`: Verify coordination with 2+ Mediators
7. `test_memory_pool_integration`: Verify shared memory communication
8. `test_qcpp_integration`: Verify QCPP adapter shared between agents

#### test_protein.py Integration (`test_protein_integration.py`)

**Test Coverage**:
1. `test_geometric_analysis_invocation`: Verify module called during protein test
2. `test_results_json_format`: Verify geometric data in output JSON
3. `test_mediator_flag`: Verify --enable-mediators flag works
4. `test_error_handling`: Verify test continues if geometric analysis fails
5. `test_performance_impact`: Verify total runtime increase < 10%



### Performance Tests

**Benchmarks**:

1. **Geometric Analysis Latency**:
   - Target: < 50ms for 200 residues
   - Measure: Average over 100 runs
   - Proteins: 1VII (35), 1UBQ (76), 1LYZ (129)

2. **Cache Hit Latency**:
   - Target: < 1ms
   - Measure: Average over 1000 cache hits
   - Cache size: 1000 entries

3. **Mediator Detection Cycle**:
   - Target: < 10ms per cycle
   - Measure: Average over 100 cycles
   - Configuration: All detections enabled

4. **Broadcast Latency**:
   - Target: < 5ms per broadcast
   - Measure: Average over 100 broadcasts
   - Agents: 20 exploration agents

5. **Memory Footprint**:
   - Target: < 100MB for 5000 cache entries
   - Measure: Peak memory usage
   - Monitor: Cache size, entry sizes

6. **End-to-End Impact**:
   - Target: < 10% runtime increase with Mediators
   - Measure: Total exploration time
   - Configuration: 20 agents, 200 iterations

**Performance Monitoring**:
- Log timing metrics for all major operations
- Track cache hit rates (target: > 80%)
- Monitor memory usage trends
- Alert on performance degradation



## Implementation Details

### Geometric Attractor Module

#### Golden Ratio Pattern Detection

**Algorithm**:
```
1. Extract CA coordinates from conformation
2. Calculate pairwise distance matrix (n×n)
3. Sample distance ratios intelligently:
   - For each residue i:
     - Compare distances to next 10 neighbors (j = i+1 to i+10)
     - For each pair (i,j), compare to next 10 (k = j+1 to j+10)
     - Calculate ratio = max(d_ij, d_ik) / min(d_ij, d_ik)
     - If ratio in [φ - tolerance, φ + tolerance], count as φ pattern
4. Calculate percentage = (φ patterns / total ratios) × 100
5. Return percentage score

Complexity: O(n²) instead of O(n⁴) due to intelligent sampling
```

**Rationale**: Full O(n⁴) analysis is prohibitively expensive. Sampling local neighborhoods captures most geometric patterns while maintaining performance.

#### Platonic Solid Similarity

**Algorithm**:
```
1. Center coordinates at origin
2. Calculate principal axes via SVD
3. Compute eigenvalue distribution
4. For each Platonic solid:
   - Calculate expected symmetry properties
   - Compare protein symmetry to ideal solid
   - Score based on:
     * Size ratio (protein size vs solid complexity)
     * Rotational symmetry match
     * Face/vertex distribution
5. Return similarity score 0.0-1.0

Special handling for φ-containing solids (dodecahedron, icosahedron):
- Boost score if golden ratio patterns detected
- Weight by φ pattern percentage
```

**Rationale**: Direct coordinate comparison is impractical. Symmetry-based comparison captures essential geometric properties.

#### Caching Strategy

**Implementation**:
```python
from collections import OrderedDict
import time

class LRUCache:
    def __init__(self, max_size: int, ttl_seconds: int):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        
    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check TTL
        if time.time() - entry.timestamp > self.ttl_seconds:
            del self.cache[key]
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        entry.access_count += 1
        entry.last_access = int(time.time() * 1000)
        
        return entry.result
    
    def put(self, key: str, result: Any) -> None:
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove oldest
        
        entry = CacheEntry(
            key=key,
            result=result,
            timestamp=int(time.time() * 1000),
            access_count=0,
            last_access=int(time.time() * 1000),
            size_bytes=sys.getsizeof(result)
        )
        
        self.cache[key] = entry
```



### Mediator Agents Module

#### THz Resonance Detection

**Algorithm**:
```
1. For each conformation in detection cycle:
   - Check cache for THz signature
   - If not cached:
     * Calculate THz signature via QCPP
     * Store in cache with conformation hash
   
2. Cluster THz signatures:
   - Use spectral correlation as similarity metric
   - Apply DBSCAN clustering with eps=0.3
   - Identify clusters with size > 1
   
3. For each cluster:
   - Calculate cluster statistics
   - Identify representative conformation (centroid)
   - Compute significance = cluster_size / total_conformations
   
4. If significance > 0.1:
   - Create PatternDetection with THzResonanceData
   - Relay to QCPP for validation
   - Broadcast to exploration agents

Complexity: O(n²) for clustering, but n is small (detection cycle size)
```

**Spectral Correlation**:
```python
def calculate_spectral_correlation(freq1, int1, freq2, int2):
    """
    Calculate correlation between two THz spectra.
    
    Returns: Correlation coefficient 0.0-1.0
    """
    # Interpolate to common frequency grid
    common_freq = np.linspace(0, 10, 1000)  # 0-10 THz
    int1_interp = np.interp(common_freq, freq1, int1)
    int2_interp = np.interp(common_freq, freq2, int2)
    
    # Normalize
    int1_norm = int1_interp / np.linalg.norm(int1_interp)
    int2_norm = int2_interp / np.linalg.norm(int2_interp)
    
    # Pearson correlation
    corr = np.corrcoef(int1_norm, int2_norm)[0, 1]
    
    return max(0.0, corr)  # Clamp to [0, 1]
```

#### Folding Dynamics Detection

**Algorithm**:
```
1. Extract phi-psi angles from conformation
2. Classify each residue:
   - Helix: phi in [-70, -50], psi in [-50, -30]
   - Sheet: phi in [-140, -100], psi in [100, 140]
   - Turn: Specific turn type patterns (I, II, III, IV)
   - Coil: Everything else
   
3. Identify continuous regions:
   - Helix: ≥ 4 consecutive helix residues
   - Sheet: ≥ 3 consecutive sheet residues
   - Turn: 3-4 residues with turn geometry
   
4. Calculate percentages:
   - helix_pct = (helix_residues / total_residues) × 100
   - sheet_pct = (sheet_residues / total_residues) × 100
   - turn_pct = (turn_residues / total_residues) × 100
   - coil_pct = 100 - helix_pct - sheet_pct - turn_pct
   
5. Compute significance:
   - High if helix_pct > 30% or sheet_pct > 20%
   - Medium if helix_pct > 15% or sheet_pct > 10%
   - Low otherwise
   
6. If significance ≥ medium:
   - Create PatternDetection with FoldingDynamicsData
   - Broadcast to exploration agents

Complexity: O(n) where n = number of residues
```



#### Geometric Similarity Detection

**Algorithm**:
```
1. Maintain reference conformations:
   - Store best conformation from each agent
   - Store conformations with high geometric scores
   - Limit to 100 references (memory constraint)
   
2. For each new conformation:
   - Calculate RMSD to each reference
   - If RMSD < threshold (default 2.0 Å):
     * Mark as geometrically similar
     * Invoke Geometric Attractor Module
     * Compare geometric patterns
   
3. Calculate structural overlap:
   - Count residues within 2.0 Å of reference
   - overlap_pct = (matching_residues / total_residues) × 100
   
4. Compute significance:
   - High if RMSD < 1.0 Å and overlap > 80%
   - Medium if RMSD < 2.0 Å and overlap > 60%
   - Low otherwise
   
5. If significance ≥ medium:
   - Create PatternDetection with GeometricSimilarityData
   - Relay to QCPP for validation
   - Broadcast to exploration agents

Complexity: O(m × n) where m = references, n = residues
Optimization: Use spatial hashing to reduce m
```

#### Broadcast Throttling

**Implementation**:
```python
class BroadcastThrottler:
    def __init__(self, max_rate: int = 10):
        self.max_rate = max_rate  # Messages per second
        self.window_size = 1.0    # 1 second window
        self.message_times = []
        
    def can_broadcast(self) -> bool:
        current_time = time.time()
        
        # Remove messages outside window
        self.message_times = [
            t for t in self.message_times 
            if current_time - t < self.window_size
        ]
        
        # Check rate
        if len(self.message_times) < self.max_rate:
            self.message_times.append(current_time)
            return True
        
        return False
    
    def prioritize_message(self, pattern: PatternDetection) -> bool:
        """
        Decide if message should be sent based on significance.
        High-significance messages bypass throttle.
        """
        if pattern.significance > 0.8:
            return True  # Always send high-significance
        
        return self.can_broadcast()
```



## Design Decisions and Rationales

### Decision 1: Separate Geometric Attractor Module

**Decision**: Create standalone module instead of integrating into existing components

**Rationale**:
- **Modularity**: Can be used independently by test_protein.py and Mediator Agents
- **Testing**: Easier to test in isolation
- **Reusability**: Other tools can import and use the module
- **Maintenance**: Clear separation of concerns

**Trade-offs**:
- Additional file/module overhead
- Need to pass dependencies explicitly

**Conclusion**: Benefits outweigh costs for long-term maintainability

### Decision 2: Mediator Agents as IProteinAgent Implementation

**Decision**: Implement Mediators as full agents rather than utility classes

**Rationale**:
- **Consistency**: Follows existing UBF architecture patterns
- **Integration**: Works seamlessly with MultiAgentCoordinator
- **Consciousness**: Can maintain their own consciousness state
- **Flexibility**: Can be extended with additional behaviors

**Trade-offs**:
- More complex than simple utility class
- Requires implementing full IProteinAgent interface

**Conclusion**: Architectural consistency is worth the complexity

### Decision 3: Shared Memory for Pattern Broadcasting

**Decision**: Use existing SharedMemoryPool instead of direct agent communication

**Rationale**:
- **Existing Infrastructure**: Leverages proven UBF component
- **Scalability**: Handles 100+ agents efficiently
- **Decoupling**: Mediators don't need references to all agents
- **Filtering**: Agents can selectively consume relevant patterns

**Trade-offs**:
- Indirect communication (slight latency)
- All agents see all patterns (filtering overhead)

**Conclusion**: Existing infrastructure is optimal solution

### Decision 4: LRU Cache with TTL

**Decision**: Implement LRU eviction with time-to-live expiration

**Rationale**:
- **Memory Bounds**: LRU prevents unbounded growth
- **Freshness**: TTL ensures stale data is removed
- **Performance**: O(1) access with OrderedDict
- **Simplicity**: Well-understood caching strategy

**Trade-offs**:
- May evict useful entries under memory pressure
- TTL adds complexity to cache logic

**Conclusion**: Standard approach balances all concerns

### Decision 5: Intelligent Distance Ratio Sampling

**Decision**: Sample local neighborhoods instead of all pairs

**Rationale**:
- **Performance**: O(n²) vs O(n⁴) is critical for large proteins
- **Accuracy**: Local patterns capture most geometric information
- **Scalability**: Enables analysis of 200+ residue proteins

**Trade-offs**:
- May miss some long-range φ patterns
- Sampling strategy affects results

**Conclusion**: Performance requirement necessitates sampling

### Decision 6: Optional Mediator Agents

**Decision**: Make Mediators opt-in via configuration flag

**Rationale**:
- **Backward Compatibility**: Existing workflows unchanged
- **Performance**: Users can disable if not needed
- **Experimentation**: Easy to compare with/without Mediators
- **Gradual Adoption**: Users can enable when ready

**Trade-offs**:
- Additional configuration complexity
- Need to test both modes

**Conclusion**: Flexibility is essential for production system



## Security and Privacy Considerations

### Data Handling

**Protein Sequences**:
- No PII or sensitive data in protein sequences
- All data is scientific/public domain
- No encryption required

**Cache Storage**:
- In-memory only (no disk persistence)
- Cleared on process termination
- No sensitive information cached

**Logging**:
- Log only aggregate statistics
- No individual conformation data in logs
- Performance metrics only

### Resource Limits

**Memory Protection**:
- Hard limit: 100 MB for geometric cache
- Hard limit: 200 MB for Mediator caches
- Automatic eviction prevents OOM

**CPU Protection**:
- Timeout: 100ms for geometric analysis
- Timeout: 50ms for pattern detection
- Prevents infinite loops or hangs

**Disk Protection**:
- No disk writes except results JSON
- Results directory size not limited (user responsibility)

### Dependency Security

**External Libraries**:
- NumPy: Trusted, widely used
- BioPython: Trusted, maintained
- SciPy: Trusted, widely used
- All dependencies already in project

**No Network Access**:
- All computation local
- No external API calls
- No data transmission



## Deployment and Configuration

### Installation

**No Additional Dependencies**:
- All required libraries already in requirements.txt
- No new external dependencies needed

**File Structure**:
```
ubf_protein/
├── geometric_attractor.py          # New: Geometric Attractor Module
├── mediator_agent.py                # New: Mediator Agent implementation
├── mediator_config.py               # New: Configuration classes
├── tests/
│   ├── test_geometric_attractor.py  # New: Unit tests
│   ├── test_mediator_agent.py       # New: Unit tests
│   └── test_mediator_integration.py # New: Integration tests
└── examples/
    └── mediator_example.py          # New: Usage example

test_protein.py                      # Modified:


## Deployment and Configuration

### Installation

**No Additional Dependencies**:
- All required libraries already in project (NumPy, BioPython, SciPy)
- Pure Python implementation (PyPy compatible)
- No compilation required

**File Structure**:
```
ubf_protein/
├── geometric_attractor.py          # New: Geometric Attractor Module
├── mediator_agent.py                # New: Mediator Agent implementation
├── mediator_config.py               # New: Configuration classes
├── pattern_detection.py             # New: Pattern detection data models
├── multi_agent_coordinator.py       # Modified: Add Mediator support
└── tests/
    ├── test_geometric_attractor.py  # New: Unit tests
    ├── test_mediator_agent.py       # New: Unit tests
    └── test_mediator_integration.py # New: Integration tests

test_protein.py                      # Modified: Add geometric analysis v2
```

### Configuration Examples

**Default Configuration** (Mediators Disabled):
```bash
python test_protein.py --pdb 1UBQ
# Uses existing workflow, no Mediators
```

**Enable Mediators**:
```bash
python test_protein.py --pdb 1UBQ --enable-mediators
# Enables 2 Mediator Agents with default settings
```

**Custom Mediator Count**:
```bash
python test_protein.py --pdb 1UBQ --enable-mediators --mediator-count 4
# Enables 4 Mediator Agents for larger proteins
```

**Programmatic Configuration**:
```python
from ubf_protein.mediator_config import MediatorConfig

config = MediatorConfig(
    thz_similarity_threshold=0.8,      # Stricter THz matching
    geometric_similarity_threshold=1.5, # Stricter geometric matching
    relay_frequency=10,                 # More frequent detection
    cache_size=20000,                   # Larger cache
    enable_thz_detection=True,
    enable_folding_detection=True,
    enable_geometric_detection=True
)

coordinator = MultiAgentCoordinator(
    protein_sequence=sequence,
    enable_mediators=True,
    mediator_count=3,
    mediator_config=config
)
```

### Performance Tuning

**For Small Proteins (< 50 residues)**:
```python
config = MediatorConfig(
    relay_frequency=10,      # More frequent detection
    cache_size=5000,         # Smaller cache
    enable_thz_detection=True,
    enable_folding_detection=True,
    enable_geometric_detection=False  # Less useful for small proteins
)
```

**For Large Proteins (> 150 residues)**:
```python
config = MediatorConfig(
    relay_frequency=30,      # Less frequent (performance)
    cache_size=20000,        # Larger cache
    enable_thz_detection=False,  # Expensive for large proteins
    enable_folding_detection=True,
    enable_geometric_detection=True
)
```

**For High-Accuracy Mode**:
```python
config = MediatorConfig(
    thz_similarity_threshold=0.9,      # Very strict
    geometric_similarity_threshold=1.0, # Very strict
    relay_frequency=5,                  # Very frequent
    cache_size=50000,                   # Very large cache
)
```



## Monitoring and Observability

### Metrics to Track

**Geometric Attractor Module**:
- Analysis count (total invocations)
- Cache hit rate (percentage)
- Average analysis time (milliseconds)
- Peak analysis time (milliseconds)
- Cache size (entries)
- Memory usage (MB)

**Mediator Agents**:
- Pattern detections by type (THz, folding, geometric)
- Detection significance distribution
- Broadcast count (total messages sent)
- Broadcast throttle events (messages dropped)
- QCPP relay count (validations requested)
- Cache hit rate per pattern type
- Average detection time per type

**System Impact**:
- Total runtime with/without Mediators
- Memory overhead (MB)
- Exploration agent performance impact
- Shared memory pool utilization

### Logging Strategy

**Log Levels**:
- **DEBUG**: Cache operations, individual detections
- **INFO**: Pattern broadcasts, statistics summaries
- **WARNING**: Performance degradation, cache evictions, throttling
- **ERROR**: QCPP failures, cache corruption, invalid configurations

**Log Format**:
```
[TIMESTAMP] [LEVEL] [MODULE] [AGENT_ID] Message
```

**Example Logs**:
```
[2025-01-15 10:23:45] [INFO] [GeometricAttractor] Analysis complete: 45.2ms, cache_hit=False
[2025-01-15 10:23:46] [INFO] [MediatorAgent] [M1] THz resonance detected: cluster_size=5, significance=0.82
[2025-01-15 10:23:46] [WARNING] [MediatorAgent] [M2] Broadcast throttled: rate=12/s exceeds limit
[2025-01-15 10:23:47] [ERROR] [MediatorAgent] [M1] QCPP relay failed: timeout after 100ms
```

### Health Checks

**Geometric Attractor Health**:
```python
def check_geometric_health(analyzer: GeometricAttractorAnalyzer) -> Dict[str, str]:
    stats = analyzer.get_cache_stats()
    
    health = {
        'status': 'healthy',
        'issues': []
    }
    
    # Check cache hit rate
    if stats['cache_hit_rate'] < 50:
        health['issues'].append('Low cache hit rate')
        health['status'] = 'degraded'
    
    # Check analysis time
    if stats['avg_analysis_time_ms'] > 75:
        health['issues'].append('Slow analysis time')
        health['status'] = 'degraded'
    
    # Check memory usage
    if stats['memory_mb'] > 150:
        health['issues'].append('High memory usage')
        health['status'] = 'degraded'
    
    return health
```

**Mediator Agent Health**:
```python
def check_mediator_health(mediator: MediatorAgent) -> Dict[str, str]:
    stats = mediator.get_detection_statistics()
    
    health = {
        'status': 'healthy',
        'issues': []
    }
    
    # Check detection rate
    if stats['detections_per_cycle'] < 0.1:
        health['issues'].append('Low detection rate')
        health['status'] = 'degraded'
    
    # Check broadcast success
    if stats['broadcast_throttle_rate'] > 0.2:
        health['issues'].append('High throttle rate')
        health['status'] = 'degraded'
    
    # Check QCPP relay failures
    if stats['qcpp_failure_rate'] > 0.1:
        health['issues'].append('QCPP relay issues')
        health['status'] = 'degraded'
    
    return health
```



## Migration and Backward Compatibility

### Existing Code Impact

**test_protein.py**:
- **Breaking Changes**: None
- **New Features**: Optional --enable-mediators flag
- **Deprecations**: None (existing geometric analysis remains)
- **Migration Path**: Gradual adoption, old code continues to work

**MultiAgentCoordinator**:
- **Breaking Changes**: None
- **New Parameters**: Optional enable_mediators, mediator_count, mediator_config
- **Default Behavior**: Unchanged (Mediators disabled by default)
- **Migration Path**: Existing code works without modification

**Shared Memory Pool**:
- **Breaking Changes**: None
- **New Message Types**: PatternDetection messages
- **Filtering**: Agents ignore unknown message types
- **Migration Path**: Transparent to existing agents

### Deprecation Strategy

**Old Geometric Analysis** (test_geometric_attractors.py):
- **Status**: Remains available for research
- **Recommendation**: Use new GeometricAttractorAnalyzer for production
- **Timeline**: No removal planned, both coexist
- **Documentation**: Mark as "experimental" vs "production-ready"

### Version Compatibility

**Minimum Requirements**:
- Python 3.8+ (existing requirement)
- NumPy 1.20.0+ (existing requirement)
- BioPython 1.79+ (existing requirement)
- SciPy 1.7.0+ (existing requirement)

**No New Dependencies**: All features use existing libraries



## Future Enhancements

### Phase 2 Enhancements (Post-Initial Release)

**1. Advanced Pattern Detection**:
- Hydrogen bond network analysis
- Hydrophobic core formation detection
- Salt bridge pattern recognition
- Disulfide bond formation tracking

**2. Machine Learning Integration**:
- Learn optimal detection thresholds from successful runs
- Predict pattern significance using historical data
- Adaptive relay frequency based on protein characteristics

**3. Visualization**:
- Real-time pattern detection dashboard
- Geometric attractor heatmaps
- THz resonance cluster visualization
- Folding dynamics timeline

**4. Distributed Computing**:
- Multi-node Mediator coordination
- Distributed pattern cache
- Cross-system pattern sharing

**5. Advanced Caching**:
- Persistent cache (disk-based)
- Cache warming from previous runs
- Predictive cache preloading

### Research Opportunities

**1. Geometric Attractor Hypothesis Validation**:
- Correlate φ patterns with folding success
- Test Platonic solid guidance effectiveness
- Quantify geometric attractor influence

**2. THz Signature Analysis**:
- Validate THz clustering with experimental data
- Correlate resonance patterns with stability
- Develop THz-based quality metrics

**3. Multi-Agent Coordination**:
- Optimize Mediator count for different protein sizes
- Study information flow patterns
- Measure collective intelligence benefits

**4. Performance Optimization**:
- GPU acceleration for geometric calculations
- Parallel pattern detection
- Advanced sampling strategies



## Summary

This design document describes a comprehensive architecture for two new modules in the QCPP-UBF platform:

### Geometric Attractor Module
- **Purpose**: Production-ready geometric pattern analysis with percentage scores
- **Performance**: < 50ms analysis, < 1ms cache hits, < 100MB memory
- **Integration**: Seamless with test_protein.py and Mediator Agents
- **Output**: Structured results for φ patterns, Platonic similarities, symmetry metrics

### Mediator Agents Module
- **Purpose**: Specialized agents for pattern detection and information relay
- **Capabilities**: THz resonance, folding dynamics, geometric similarity detection
- **Architecture**: Full IProteinAgent implementation with consciousness coordinates
- **Communication**: Shared memory pool for broadcasting to exploration agents
- **Performance**: < 10ms detection cycles, efficient caching, throttled broadcasts

### Key Design Principles
1. **Modularity**: Clean separation of concerns with well-defined interfaces
2. **Performance**: Optimized algorithms and caching for real-time operation
3. **Compatibility**: Zero breaking changes, optional features
4. **Scalability**: Supports 100+ agents and large proteins (200+ residues)
5. **Maintainability**: Comprehensive testing, logging, and monitoring

### Implementation Readiness
- All interfaces defined
- Algorithms specified with complexity analysis
- Error handling strategy documented
- Testing strategy comprehensive (85%+ coverage target)
- Performance benchmarks established
- Migration path clear

The design is ready for implementation with all requirements addressed and technical decisions documented.

