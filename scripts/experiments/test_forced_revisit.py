"""Force a revisit scenario to verify caching works"""
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
from ubf_protein.qcpp_config import get_default_config
from ubf_protein.models import Conformation
import time

# Create coordinator
adapter = QCPPIntegrationAdapter(get_default_config())
coord = MultiAgentCoordinator(
    protein_sequence='ACDEFGH',
    qcpp_integration=adapter,
    qcpp_analysis_frequency=1
)
coord.initialize_agents(count=1, diversity_profile='balanced')

agent = coord._agents[0]

# Get initial conformation
state = agent._current_conformation
coords = state.atom_coordinates[:]  # Copy coordinates

# Create identical conformation manually
conf_copy = Conformation(
    sequence=state.sequence,
    atom_coordinates=coords,  # Exact same coordinates
    secondary_structure=state.secondary_structure
)

print("Testing QCPP caching with identical conformations...")
print(f"Original hash: {coord._hash_conformation(state)[:16]}...")
print(f"Copy hash:     {coord._hash_conformation(conf_copy)[:16]}...")
print(f"Hashes match: {coord._hash_conformation(state) == coord._hash_conformation(conf_copy)}")

# Analyze original
print("\n1. Analyzing ORIGINAL conformation...")
start = time.time()
metrics1 = adapter.analyze_conformation(state)
time1 = (time.time() - start) * 1000
print(f"   Time: {time1:.2f}ms")

# Store in registry
coord.store_qcpp_in_registry(state, metrics1)
print(f"   Stored in registry")

# Try to retrieve from registry
print("\n2. Querying registry for IDENTICAL conformation...")
start = time.time()
metrics2 = coord.get_qcpp_from_registry(conf_copy)
time2 = (time.time() - start) * 1000
print(f"   Time: {time2:.3f}ms")
print(f"   Found in registry: {metrics2 is not None}")

if metrics2:
    print(f"   ✓ Cache HIT! Speedup: {time1/time2:.0f}×")
    print(f"   QCP scores match: {metrics1.qcp_score == metrics2.qcp_score}")
else:
    print(f"   ✗ Cache MISS - something is wrong!")

# Check stats
stats = coord.get_registry_stats()
print(f"\n3. Registry stats:")
print(f"   Total queries: {stats['total_queries']}")
print(f"   Cache hits: {stats['cache_hits']}")
print(f"   Hit rate: {stats['hit_rate_percent']:.1f}%")
