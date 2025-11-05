"""Quick test to verify registry wiring"""
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
from ubf_protein.qcpp_config import get_default_config

# Create coordinator with QCPP
adapter = QCPPIntegrationAdapter(get_default_config())
coord = MultiAgentCoordinator(
    protein_sequence='ACDEFGH',
    qcpp_integration=adapter,
    qcpp_analysis_frequency=1  # Analyze every iteration for testing
)
coord.initialize_agents(count=2, diversity_profile='balanced')

# Check wiring
agents = coord._agents
print(f"✓ Created {len(agents)} agents")
print(f"✓ Coordinator has registry dict: {hasattr(coord, '_global_qcpp_registry')}")
print(f"✓ Registry is dict: {isinstance(coord._global_qcpp_registry, dict)}")
print(f"✓ Registry size: {len(coord._global_qcpp_registry)}")
print(f"✓ Coordinator has get method: {hasattr(coord, 'get_qcpp_from_registry')}")
print(f"✓ Coordinator has store method: {hasattr(coord, 'store_qcpp_in_registry')}")
print(f"✓ Coordinator has stats method: {hasattr(coord, 'get_registry_stats')}")

# Test a longer run to see if registry gets reused
print("\nRunning 50 iterations per agent (100 total conformations)...")
coord2 = MultiAgentCoordinator(
    protein_sequence='ACDEFGH',
    qcpp_integration=adapter,
    qcpp_analysis_frequency=5  # Every 5 iterations
)
coord2.initialize_agents(count=5, diversity_profile='balanced')
coord2.run_parallel_exploration(iterations=50)

print(f"\n✓ Registry size after 50x5 iterations: {len(coord2._global_qcpp_registry)}")
stats2 = coord2.get_registry_stats()
print(f"✓ Registry stats:")
for key, value in stats2.items():
    print(f"   {key}: {value}")
