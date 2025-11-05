# Design Document: Large-Scale Protein Validation Framework

## Overview

This design document outlines the architecture for a comprehensive validation framework that tests 50-75 proteins using the integrated QCPP-UBF platform. The system extends the existing `ValidationSuite` class with new components for protein selection, phased testing, progress tracking, statistical analysis, and automated research documentation.

### Design Goals

1. **Scalability**: Handle 50-75 proteins efficiently with batch execution and resource management
2. **Reproducibility**: Capture all parameters, seeds, and configurations for independent verification
3. **Iterative Research**: Support phased testing with quality gates and parameter adjustment between phases
4. **Automation**: Minimize manual work through automated data collection, analysis, and documentation
5. **Integration**: Leverage existing ValidationSuite, MultiAgentCoordinator, and QCPP integration

## Architecture

### High-Level Component Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                  LargeScaleValidationCampaign                   │
│  (Orchestrates entire 50-75 protein validation campaign)       │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├──► ProteinSelector
             │    (Curates diverse test set from PDB)
             │
             ├──► PhaseManager
             │    (Manages 4-phase testing with quality gates)
             │
             ├──► BatchExecutor
             │    (Parallel execution with resource management)
             │
             ├──► ResultsRepository
             │    (Centralized storage: JSON + Markdown)
             │
             ├──► ProgressTracker
             │    (Real-time monitoring and dashboards)
             │
             ├──► StatisticalAnalyzer
             │    (Correlation analysis and pattern detection)
             │
             ├──► FailureAnalyzer
             │    (Detailed analysis of failed predictions)
             │
             └──► DocumentationGenerator
                  (Auto-generate research reports and figures)
```

### Integration with Existing Systems

The framework builds on existing components:
- **ValidationSuite**: Core validation logic (RMSD, GDT-TS, TM-score)
- **MultiAgentCoordinator**: Protein structure prediction engine
- **QCPPIntegrationAdapter**: Quantum physics feedback
- **NativeStructureLoader**: PDB structure loading
- **CheckpointManager**: Checkpoint/resume functionality

## Components and Interfaces

### 1. ProteinSelector

**Purpose**: Systematically select 50-75 diverse proteins from PDB database

**Key Methods**:
```python
class ProteinSelector:
    def select_proteins(self, target_count: int = 60) -> List[ProteinMetadata]
    def filter_by_size(self, proteins: List, size_categories: Dict) -> List
    def filter_by_structural_class(self, proteins: List) -> List
    def filter_by_resolution(self, proteins: List, max_resolution: float = 2.5) -> List
    def filter_by_completeness(self, proteins: List, max_missing_pct: float = 10.0) -> List
    def export_selection(self, output_path: str) -> None
```

**Data Model**:
```python
@dataclass
class ProteinMetadata:
    pdb_id: str
    sequence_length: int
    size_category: str  # tiny, small, medium, large
    structural_class: str  # all-alpha, all-beta, alpha-beta, alpha+beta, irregular
    experimental_method: str  # X-ray, NMR
    resolution: Optional[float]  # Angstroms (None for NMR)
    missing_residues_pct: float
    organism: str
    description: str
```

**Selection Strategy**:
- Query PDB database or use curated list
- Apply filters: size, structural class, resolution, completeness
- Balance distribution across categories
- Prioritize well-studied proteins for Phase 1

### 2. PhaseManager

**Purpose**: Organize testing into 4 progressive phases with quality gates

**Key Methods**:
```python
class PhaseManager:
    def initialize_phases(self, proteins: List[ProteinMetadata]) -> Dict[int, Phase]
    def get_current_phase(self) -> Phase
    def advance_to_next_phase(self) -> bool
    def check_quality_gate(self, phase: Phase) -> QualityGateResult
    def generate_phase_summary(self, phase: Phase) -> PhaseSummaryReport
    def allow_parameter_adjustment(self) -> Dict[str, Any]
```

**Data Model**:
```python
@dataclass
class Phase:
    phase_number: int  # 1, 2, 3, 4
    protein_count: int  # 10, 15, 25, remaining
    proteins: List[ProteinMetadata]
    status: str  # pending, in_progress, completed, failed_gate
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    success_rate: float
    average_rmsd: float
    average_gdt_ts: float

@dataclass
class QualityGateResult:
    passed: bool
    success_rate: float
    threshold: float  # 60% for Phase 1
    issues_identified: List[str]
    recommendations: List[str]
```

**Phase Distribution**:
- Phase 1: 10 proteins (easy, well-studied, high resolution)
- Phase 2: 15 proteins (mixed difficulty)
- Phase 3: 25 proteins (diverse characteristics)
- Phase 4: Remaining proteins (challenging cases)

### 3. BatchExecutor

**Purpose**: Execute multiple test runs efficiently with resource management

**Key Methods**:
```python
class BatchExecutor:
    def execute_batch(self, proteins: List[ProteinMetadata], 
                     max_parallel: int = 3) -> List[ValidationReport]
    def monitor_resources(self) -> ResourceMetrics
    def estimate_completion_time(self, remaining: int) -> timedelta
    def checkpoint_batch(self, checkpoint_name: str) -> str
    def resume_batch(self, checkpoint_file: str) -> None
    def prioritize_by_size(self, proteins: List[ProteinMetadata]) -> List
```

**Data Model**:
```python
@dataclass
class ResourceMetrics:
    cpu_usage_percent: float
    memory_usage_mb: float
    disk_usage_mb: float
    active_processes: int
    throttle_recommended: bool

@dataclass
class BatchProgress:
    total_proteins: int
    completed: int
    in_progress: int
    pending: int
    failed: int
    estimated_completion: datetime
    average_time_per_protein: float
```

**Execution Strategy**:
- Prioritize small proteins first (faster execution)
- Limit parallelization to 3-5 concurrent tests (resource constraints)
- Monitor CPU/memory and throttle if needed
- Checkpoint every 5 completed tests
- Resume from checkpoint on interruption

### 4. ResultsRepository

**Purpose**: Centralized storage for all test results and metadata

**Key Methods**:
```python
class ResultsRepository:
    def store_result(self, report: ValidationReport, metadata: TestRunMetadata) -> None
    def append_to_markdown(self, report: ValidationReport) -> None
    def save_to_json_database(self, report: ValidationReport) -> None
    def save_predicted_structure(self, pdb_id: str, structure: Conformation) -> str
    def get_all_results(self) -> List[ValidationReport]
    def query_results(self, filters: Dict) -> List[ValidationReport]
    def export_logs(self, pdb_id: str) -> str
```

**Data Model**:
```python
@dataclass
class TestRunMetadata:
    pdb_id: str
    timestamp: datetime
    software_version: str
    python_version: str
    num_agents: int
    iterations_per_agent: int
    qcpp_enabled: bool
    random_seed: int
    adaptive_config: Dict[str, Any]
    execution_parameters: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
```

**Storage Structure**:
```
results/
├── COMPREHENSIVE_TEST_RESULTS.md  # Human-readable results
├── validation_database.json        # Machine-readable database
├── logs/
│   ├── 1UBQ_20250126_143022.log
│   └── 1CRN_20250126_144530.log
├── structures/
│   ├── 1UBQ_predicted_20250126_143022.pdb
│   └── 1CRN_predicted_20250126_144530.pdb
└── metadata/
    ├── 1UBQ_metadata.json
    └── 1CRN_metadata.json
```

### 5. ProgressTracker

**Purpose**: Real-time monitoring and visualization of testing progress

**Key Methods**:
```python
class ProgressTracker:
    def update_progress(self, report: ValidationReport) -> None
    def get_dashboard_data(self) -> DashboardData
    def generate_interim_report(self, phase: Phase) -> InterimReport
    def calculate_running_averages(self) -> RunningAverages
    def identify_outliers(self, threshold_std: float = 2.0) -> List[str]
    def generate_comparison_visualization(self, pdb_id: str) -> str
```

**Data Model**:
```python
@dataclass
class DashboardData:
    phase: int
    completed_tests: int
    pending_tests: int
    success_rate: float
    running_avg_rmsd: float
    running_avg_gdt_ts: float
    running_avg_tm_score: float
    running_avg_energy: float
    outliers: List[str]
    recent_completions: List[str]

@dataclass
class InterimReport:
    phase: int
    completion_percentage: float
    tests_completed: int
    tests_remaining: int
    current_success_rate: float
    trends: Dict[str, str]  # improving, stable, declining
    issues_detected: List[str]
```

**Visualization Outputs**:
- Progress dashboard (JSON for web display)
- Structure comparison images (predicted vs native)
- Metric distribution plots (RMSD, GDT-TS, energy)

### 6. StatisticalAnalyzer

**Purpose**: Automated statistical analysis and pattern detection

**Key Methods**:
```python
class StatisticalAnalyzer:
    def calculate_correlations(self, results: List[ValidationReport]) -> CorrelationMatrix
    def compare_size_categories(self, results: List[ValidationReport]) -> StatisticalComparison
    def generate_distribution_plots(self, results: List[ValidationReport]) -> List[str]
    def identify_predictive_features(self, results: List[ValidationReport]) -> FeatureImportance
    def calculate_confidence_intervals(self, results: List[ValidationReport]) -> ConfidenceIntervals
```

**Data Model**:
```python
@dataclass
class CorrelationMatrix:
    size_vs_rmsd: float
    size_vs_gdt_ts: float
    secondary_structure_vs_rmsd: float
    resolution_vs_accuracy: float
    correlations: Dict[str, float]

@dataclass
class StatisticalComparison:
    category_means: Dict[str, float]
    category_stds: Dict[str, float]
    p_values: Dict[str, float]  # Statistical significance
    effect_sizes: Dict[str, float]

@dataclass
class FeatureImportance:
    features: List[str]
    importance_scores: List[float]
    predictive_power: Dict[str, float]
```

**Analysis Techniques**:
- Pearson correlation coefficients
- ANOVA for category comparisons
- Distribution fitting (normal, log-normal)
- Confidence intervals (95%)
- Feature importance ranking

### 7. FailureAnalyzer

**Purpose**: Detailed analysis of failed predictions for system improvement

**Key Methods**:
```python
class FailureAnalyzer:
    def classify_failure(self, report: ValidationReport) -> FailureClassification
    def extract_common_characteristics(self, failures: List[ValidationReport]) -> FailurePatterns
    def generate_failure_visualizations(self, pdb_id: str) -> List[str]
    def analyze_energy_trajectory(self, pdb_id: str) -> TrajectoryAnalysis
    def recommend_parameter_adjustments(self, patterns: FailurePatterns) -> List[str]
```

**Data Model**:
```python
@dataclass
class FailureClassification:
    pdb_id: str
    failure_type: str  # high_rmsd, poor_energy, low_gdt_ts
    severity: str  # minor, moderate, severe
    rmsd: float
    energy: float
    gdt_ts: float

@dataclass
class FailurePatterns:
    common_size_category: Optional[str]
    common_structural_class: Optional[str]
    average_secondary_structure_content: Dict[str, float]
    common_issues: List[str]

@dataclass
class TrajectoryAnalysis:
    stuck_in_local_minima: bool
    minima_count: int
    escape_attempts: int
    energy_variance: float
    convergence_achieved: bool
```

**Failure Criteria**:
- RMSD > 8.0 Å (severe failure)
- Energy > 0 kcal/mol (unstable structure)
- GDT-TS < 30 (incorrect fold)

### 8. DocumentationGenerator

**Purpose**: Automatically generate research documentation and figures

**Key Methods**:
```python
class DocumentationGenerator:
    def generate_phase_report(self, phase: Phase, results: List[ValidationReport]) -> str
    def generate_publication_figures(self, results: List[ValidationReport]) -> List[str]
    def generate_methods_section(self, metadata: TestRunMetadata) -> str
    def generate_supplementary_tables(self, results: List[ValidationReport]) -> str
    def export_for_plotting_tools(self, results: List[ValidationReport], 
                                  formats: List[str]) -> Dict[str, str]
```

**Data Model**:
```python
@dataclass
class ResearchReport:
    title: str
    methodology: str
    results_summary: str
    statistical_analysis: str
    figures: List[str]
    tables: List[str]
    conclusions: List[str]
    generated_timestamp: datetime
```

**Generated Outputs**:
- Phase summary reports (Markdown)
- Publication-ready figures (PNG, SVG)
- Methods section (Markdown)
- Supplementary data tables (CSV, Excel)
- Plotting tool exports (CSV, JSON)

### 9. VibrationalAnalyzer

**Purpose**: Calculate real THz vibrational modes using normal mode analysis

**Key Methods**:
```python
class VibrationalAnalyzer:
    def calculate_hessian(self, conformation: Conformation) -> np.ndarray
    def get_normal_modes(self, conformation: Conformation, 
                        n_modes: int = 20) -> Tuple[np.ndarray, np.ndarray]
    def compute_thz_spectrum(self, conformation: Conformation) -> THzSpectrum
    def calculate_mode_intensities(self, modes: np.ndarray) -> np.ndarray
    def match_signature(self, spectrum: THzSpectrum, 
                       target: THzSpectrum) -> float
```

**Data Model**:
```python
@dataclass
class THzSpectrum:
    frequencies: np.ndarray  # THz frequencies (10-50 THz range)
    intensities: np.ndarray  # Mode intensities
    eigenvectors: np.ndarray  # Vibrational eigenvectors
    energy: float  # Conformation energy
    rmsd: Optional[float]  # RMSD to native (if known)
    timestamp: datetime

@dataclass
class NormalModeData:
    mode_index: int
    frequency_thz: float
    eigenvector: np.ndarray
    participation_ratio: float  # Measure of mode delocalization
```

**Implementation Details**:
- Build Hessian matrix from molecular mechanics force field
- Use scipy.linalg.eigh for eigenvalue decomposition
- Convert eigenvalues to THz: ω = √(λ) / (2π × c)
- Focus on 10-50 THz range (protein collective motions)
- Cache Hessian calculations for performance

### 10. SignatureDatabase

**Purpose**: Store and query THz signatures mapped to fold quality

**Key Methods**:
```python
class SignatureDatabase:
    def add_signature(self, signature: THzSpectrum, 
                     quality_metrics: QualityMetrics) -> None
    def query_similar_signatures(self, query: THzSpectrum, 
                                top_k: int = 10) -> List[SignatureMatch]
    def get_successful_signatures(self, rmsd_threshold: float = 5.0) -> List[THzSpectrum]
    def cluster_signatures(self, method: str = "kmeans") -> List[SignatureCluster]
    def export_database(self, output_path: str) -> None
    def calculate_signature_statistics(self) -> SignatureStatistics
```

**Data Model**:
```python
@dataclass
class QualityMetrics:
    rmsd: float
    gdt_ts: float
    tm_score: float
    energy: float
    is_successful: bool  # RMSD < 5.0 Å

@dataclass
class SignatureMatch:
    signature: THzSpectrum
    similarity_score: float  # 0-1, higher is more similar
    quality_metrics: QualityMetrics
    pdb_id: str

@dataclass
class SignatureCluster:
    cluster_id: int
    centroid: THzSpectrum
    members: List[THzSpectrum]
    average_quality: QualityMetrics
    cluster_size: int

@dataclass
class SignatureStatistics:
    total_signatures: int
    successful_signatures: int
    average_frequency_range: Tuple[float, float]
    most_common_peaks: List[float]
    quality_correlation: float  # Correlation between signature and quality
```

**Storage Strategy**:
- Store signatures in HDF5 format for efficient array storage
- Index by PDB ID, energy, RMSD for fast queries
- Maintain separate collections for successful vs failed folds
- Update incrementally as campaign progresses

### 11. VibrationalGuidedAgent (Extension)

**Purpose**: Extend ProteinAgent to use THz signatures as consciousness targets

**Key Methods**:
```python
class VibrationalGuidedAgent(ProteinAgent):
    def __init__(self, sequence: str, 
                 vibrational_analyzer: VibrationalAnalyzer,
                 signature_database: SignatureDatabase,
                 **kwargs)
    
    def evaluate_move_with_thz(self, move: ConformationalMove) -> float
    def calculate_thz_alignment_factor(self, spectrum: THzSpectrum) -> float
    def update_consciousness_with_thz(self, spectrum: THzSpectrum) -> None
    def record_thz_at_minimum(self, conformation: Conformation) -> None
    def get_thz_learning_progress(self) -> THzLearningMetrics
```

**Data Model**:
```python
@dataclass
class THzLearningMetrics:
    signatures_encountered: int
    successful_signatures_found: int
    average_similarity_to_target: float
    convergence_rate: float  # How quickly agent learns to seek good signatures
    thz_guided_moves: int
    thz_reward_total: float

@dataclass
class THzMemory(ConformationalMemory):
    thz_spectrum: THzSpectrum
    signature_similarity: float  # Similarity to known successful signatures
    vibrational_reward: float  # Reward for matching target signatures
```

**Integration with Consciousness**:
- Add THz signature matching as 6th dimension in consciousness coordinates
- Reward moves that bring spectrum closer to successful signatures
- Share high-quality THz signatures through collective memory
- Track convergence: agents learn to seek specific THz patterns over time

### 12. LargeScaleValidationCampaign (Main Orchestrator)

**Purpose**: Coordinate all components for end-to-end validation campaign

**Key Methods**:
```python
class LargeScaleValidationCampaign:
    def __init__(self, target_protein_count: int = 60, 
                 enable_qcpp: bool = True,
                 enable_thz_analysis: bool = True,
                 max_parallel: int = 3)
    
    def setup_campaign(self) -> None
    def run_campaign(self) -> CampaignResults
    def run_phase(self, phase_number: int) -> PhaseResults
    def execute_single_test(self, protein: ProteinMetadata) -> ValidationReport
    def handle_phase_completion(self, phase: Phase) -> None
    def generate_final_report(self) -> str
```

**Workflow**:
1. Initialize campaign with target protein count
2. Select proteins using ProteinSelector
3. Organize into phases using PhaseManager
4. Initialize VibrationalAnalyzer and SignatureDatabase
5. For each phase:
   - Execute batch using BatchExecutor with VibrationalGuidedAgents
   - Record THz signatures at energy minima
   - Build signature database incrementally
   - Track progress using ProgressTracker
   - Store results using ResultsRepository
   - Check quality gate using PhaseManager
   - Generate interim report using DocumentationGenerator
   - If gate fails, analyze and adjust parameters
6. After all phases:
   - Run statistical analysis using StatisticalAnalyzer
   - Analyze THz signature patterns and correlations
   - Analyze failures using FailureAnalyzer
   - Generate final report using DocumentationGenerator

## THz Vibrational Analysis Architecture

### Conceptual Foundation

The THz vibrational analysis represents a paradigm shift: instead of simulating consciousness to guide folding, **the protein's vibrational signature IS the target**. Agents learn that successful folds occur when conformations exhibit specific THz resonance patterns.

**Key Insight**: The fold happens when and only when vibrational modes sync to specific frequencies. This is "catching the protein lying about whether it's random or not" - if folding were truly random, we wouldn't see consistent THz signatures at successful folds.

### Integration Flow

```
Agent Exploration
    ↓
Generate Move → Evaluate Conformation
    ↓
[At Energy Minimum]
    ↓
Calculate THz Spectrum (VibrationalAnalyzer)
    ↓
Query Signature Database
    ↓
Calculate Similarity to Successful Signatures
    ↓
Update Consciousness Coordinates (THz dimension)
    ↓
Reward/Penalize Move Based on THz Match
    ↓
Store Signature in Database (if significant)
    ↓
Share via Collective Memory (if high quality)
```

### THz-Guided Learning Process

**Phase 1: Exploration (First 10-20% of iterations)**
- Agents explore randomly, recording THz signatures at all minima
- Build initial signature database
- No THz guidance yet (insufficient data)

**Phase 2: Pattern Recognition (20-50% of iterations)**
- Database contains enough signatures to identify patterns
- Agents begin recognizing that certain THz patterns correlate with low RMSD
- THz alignment factor starts influencing move evaluation

**Phase 3: Targeted Seeking (50-100% of iterations)**
- Agents actively seek conformations with target THz signatures
- Consciousness coordinates heavily weighted toward THz matching
- Collective memory shares successful THz patterns
- Convergence accelerates as agents "learn the protein's will"

### Performance Optimization

**Hessian Calculation Caching**:
- Hessian calculation is expensive (O(N³) for N atoms)
- Cache Hessians for similar conformations (RMSD < 0.5 Å)
- Expected cache hit rate: 20-30% during exploration
- Target: <50ms per THz spectrum calculation

**Selective THz Analysis**:
- Only calculate THz at energy minima (not every conformation)
- Use energy threshold: calculate if E < best_energy × 1.2
- Reduces THz calculations by 80-90%

**Incremental Database Updates**:
- Update signature database asynchronously
- Batch insertions every 10 signatures
- Use approximate nearest neighbor search (FAISS) for fast queries

### Validation Metrics

**THz-Specific Metrics**:
- **Signature Convergence Rate**: How quickly agents learn to seek successful signatures
- **THz-Quality Correlation**: Correlation between signature similarity and RMSD
- **Signature Cluster Count**: Number of distinct THz patterns for successful folds
- **Learning Efficiency**: Reduction in exploration time with THz guidance vs without

**Expected Outcomes**:
- **Hypothesis 1 (Strong Determinism)**: All successful folds → 1-2 THz signature clusters
- **Hypothesis 2 (Multiple Pathways)**: Successful folds → 3-5 distinct signature clusters
- **Hypothesis 3 (Stochastic)**: No clear signature clustering, weak correlation with quality

## Data Models

### Core Data Structures

```python
@dataclass
class CampaignResults:
    campaign_id: str
    start_time: datetime
    end_time: datetime
    total_proteins: int
    phases_completed: int
    overall_success_rate: float
    validation_reports: List[ValidationReport]
    phase_summaries: List[PhaseSummaryReport]
    statistical_analysis: StatisticalAnalysis
    failure_analysis: FailureAnalysis
    thz_analysis: THzAnalysisResults  # NEW
    final_report_path: str

@dataclass
class PhaseResults:
    phase_number: int
    proteins_tested: int
    success_rate: float
    average_rmsd: float
    average_gdt_ts: float
    average_energy: float
    quality_gate_passed: bool
    validation_reports: List[ValidationReport]
    interim_report_path: str
    thz_signatures_collected: int  # NEW
    thz_learning_metrics: THzLearningMetrics  # NEW

@dataclass
class THzAnalysisResults:
    total_signatures_collected: int
    successful_signature_clusters: List[SignatureCluster]
    thz_quality_correlation: float
    signature_convergence_rate: float
    average_learning_efficiency: float
    hypothesis_result: str  # "strong_determinism", "multiple_pathways", "stochastic"
    signature_database_path: str
```

## Error Handling

### Graceful Degradation

1. **PDB Download Failures**: Skip protein and log warning, continue with others
2. **Prediction Timeouts**: Set timeout (2x expected), checkpoint and move to next
3. **Resource Exhaustion**: Throttle parallelization, checkpoint, and resume
4. **Quality Gate Failures**: Pause campaign, generate analysis, await user decision
5. **File I/O Errors**: Retry 3 times, fallback to alternative storage location

### Logging Strategy

- **INFO**: Campaign progress, phase transitions, test completions
- **WARNING**: Timeouts, resource constraints, quality gate concerns
- **ERROR**: Failed predictions, I/O errors, critical failures
- **DEBUG**: Detailed execution parameters, intermediate calculations

## Testing Strategy

### Unit Tests

- ProteinSelector: Test filtering logic, distribution balance
- PhaseManager: Test phase transitions, quality gate logic
- BatchExecutor: Test prioritization, resource monitoring
- ResultsRepository: Test storage, retrieval, querying
- StatisticalAnalyzer: Test correlation calculations, statistical tests
- FailureAnalyzer: Test pattern detection, classification
- **VibrationalAnalyzer**: Test Hessian calculation, normal mode extraction, THz spectrum generation
- **SignatureDatabase**: Test signature storage, similarity queries, clustering
- **VibrationalGuidedAgent**: Test THz-guided move evaluation, consciousness updates

### Integration Tests

- End-to-end campaign with 5 test proteins
- Phase transition with quality gate failure
- Checkpoint and resume functionality
- Parallel execution with resource throttling
- **THz-guided exploration**: Test agent learning with signature database
- **Signature convergence**: Verify agents learn to seek successful THz patterns

### Validation Tests

- Compare against existing ValidationSuite results
- Verify statistical calculations against known datasets
- Validate reproducibility with same random seeds
- **THz spectrum accuracy**: Compare calculated modes against known protein vibrational data
- **Signature clustering**: Verify clustering produces meaningful groups
- **Learning efficiency**: Measure improvement in fold quality with THz guidance vs without

## Performance Considerations

### Execution Time Estimates

- Small protein (<50 residues): 2-5 minutes
- Medium protein (50-150 residues): 5-15 minutes
- Large protein (>150 residues): 15-30 minutes

**Total Campaign Time** (60 proteins, 3 parallel):
- Optimistic: 8-10 hours
- Realistic: 12-16 hours
- Conservative: 20-24 hours

### Resource Requirements

- **CPU**: 4-8 cores (3-5 parallel tests)
- **Memory**: 8-16 GB (2-3 GB per test)
- **Disk**: 5-10 GB (structures, logs, results)

### Optimization Strategies

1. **Prioritize small proteins first**: Faster early results
2. **Cache QCPP calculations**: 30-50% cache hit rate
3. **Checkpoint frequently**: Resume without data loss
4. **Adaptive parallelization**: Scale based on resource availability
5. **Incremental analysis**: Generate reports as data accumulates

## Configuration

### Campaign Configuration

```python
@dataclass
class CampaignConfig:
    target_protein_count: int = 60
    enable_qcpp: bool = True
    enable_thz_analysis: bool = True  # NEW
    max_parallel_tests: int = 3
    num_agents: int = 10
    iterations_per_agent: int = 1000
    checkpoint_interval: int = 5  # tests
    quality_gate_threshold: float = 0.60  # 60% success rate
    failure_rmsd_threshold: float = 8.0
    timeout_multiplier: float = 2.0
    random_seed: Optional[int] = None

@dataclass
class THzConfig:
    n_modes: int = 20  # Number of normal modes to calculate
    frequency_range: Tuple[float, float] = (10.0, 50.0)  # THz
    calculate_at_minima_only: bool = True
    energy_threshold_multiplier: float = 1.2  # Calculate if E < best × 1.2
    cache_hessians: bool = True
    cache_size: int = 1000
    signature_similarity_threshold: float = 0.7  # For matching
    thz_reward_weight: float = 0.3  # Weight in move evaluation
    learning_phase_threshold: float = 0.2  # Start THz guidance at 20% progress
```

### Reproducibility Configuration

```python
@dataclass
class ReproducibilityConfig:
    capture_software_versions: bool = True
    capture_random_seeds: bool = True
    capture_system_info: bool = True
    generate_reproducibility_script: bool = True
    validate_checksums: bool = True
```

## Future Enhancements

1. **Web Dashboard**: Real-time progress visualization in browser
2. **Distributed Execution**: Run tests across multiple machines
3. **Active Learning**: Prioritize proteins based on uncertainty
4. **Comparative Benchmarking**: Automated comparison with AlphaFold, RoseTTAFold
5. **Publication Export**: Direct export to LaTeX, Word formats
6. **Interactive Analysis**: Jupyter notebook integration for custom analysis
7. **Experimental THz Validation**: Compare calculated THz spectra against experimental THz spectroscopy data
8. **Transfer Learning**: Use THz signatures from one protein to guide folding of similar proteins
9. **Multi-Scale Analysis**: Combine THz (collective motions) with higher frequency modes (local vibrations)
10. **Quantum Effects**: Investigate quantum coherence in THz vibrational modes

## Summary

This design provides a comprehensive, scalable framework for validating 50-75 proteins with the QCPP-UBF platform. Key features include:

- **Systematic protein selection** across diverse characteristics
- **Phased testing** with quality gates and parameter adjustment
- **Automated data collection** to centralized repository
- **Real-time progress tracking** with dashboards and interim reports
- **Statistical analysis** for pattern detection and correlation
- **Failure analysis** for system improvement
- **Automated documentation** for research publication
- **Reproducibility** through comprehensive metadata capture
- **THz vibrational analysis** for signature-guided folding (NEW)
- **Vibrational-guided agents** that learn to seek successful THz patterns (NEW)
- **Signature database** mapping vibrational modes to fold quality (NEW)

### THz Innovation

The THz vibrational analysis represents a fundamental insight: **the protein's vibrational signature IS the consciousness target**. Instead of simulating abstract consciousness to guide folding, agents learn that successful folds occur when conformations exhibit specific THz resonance patterns in the 10-50 THz range.

This approach:
- Uses real physics (normal mode analysis) rather than synthetic approximations
- Provides falsifiable predictions about folding determinism vs stochasticity
- Enables agents to "reverse-engineer the protein's will" by seeking vibrational signatures
- Can be validated against experimental THz spectroscopy data

The architecture builds on existing ValidationSuite components while adding new capabilities for large-scale, iterative research campaigns with physics-grounded vibrational guidance.
