"""Test hash function with actual conformations"""
from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
from ubf_protein.qcpp_integration import QCPPIntegrationAdapter
from ubf_protein.qcpp_config import get_default_config

# Create coordinator with QCPP
adapter = QCPPIntegrationAdapter(get_default_config())
coord = MultiAgentCoordinator(
    protein_sequence='ACDEFGH',
    qcpp_integration=adapter,
    qcpp_analysis_frequency=1  # Every iteration
)
coord.initialize_agents(count=2, diversity_profile='balanced')

# Get initial conformations from both agents
agent1 = coord._agents[0]
agent2 = coord._agents[1]

conf1 = agent1.get_current_state()['conformation']
conf2 = agent2.get_current_state()['conformation']

# Hash them
hash1 = coord._hash_conformation(conf1)
hash2 = coord._hash_conformation(conf2)

print(f"Agent 1 conformation hash: {hash1[:16]}...")
print(f"Agent 2 conformation hash: {hash2[:16]}...")
print(f"Hashes equal: {hash1 == hash2}")

# Print first few coordinates for inspection
print(f"\nAgent 1 first 3 coords: {conf1.atom_coordinates[:3]}")
print(f"Agent 2 first 3 coords: {conf2.atom_coordinates[:3]}")

# Run 1 iteration and hash again
agent1.explore_step()
conf1_after = agent1.get_current_state()['conformation']
hash1_after = coord._hash_conformation(conf1_after)

print(f"\nAgent 1 after 1 step: {hash1_after[:16]}...")
print(f"Same as before: {hash1 == hash1_after}")

# Check what's in registry
print(f"\nRegistry size: {len(coord._global_qcpp_registry)}")
print(f"Registry keys (first 16 chars):")
for key in list(coord._global_qcpp_registry.keys())[:5]:
    print(f"  {key[:16]}...")
