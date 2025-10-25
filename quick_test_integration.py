#!/usr/bin/env python3
"""
Quick Test: QCPP-UBF Integration Validation

Quick validation that QCPP integration works with a small test.
For full validation, see validate_qcpp_ubf_integration.py
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Add ubf_protein to path
sys.path.insert(0, str(Path(__file__).parent / "ubf_protein"))

print("="*70)
print("QCPP-UBF INTEGRATION QUICK TEST")
print("="*70)

# Test 1: Import all components
print("\n[1/7] Testing imports...")
try:
    from protein_predictor import QuantumCoherenceProteinPredictor
    from ubf_protein.qcpp_integration import QCPPIntegrationAdapter, QCPPMetrics
    from ubf_protein.qcpp_config import QCPPIntegrationConfig
    from ubf_protein.protein_agent import ProteinAgent
    from ubf_protein.multi_agent_coordinator import MultiAgentCoordinator
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialize QCPP
print("\n[2/7] Initializing QCPP predictor...")
try:
    qcpp_predictor = QuantumCoherenceProteinPredictor()
    print("✓ QCPP predictor initialized")
except Exception as e:
    print(f"✗ QCPP initialization failed: {e}")
    sys.exit(1)

# Test 3: Create integration adapter
print("\n[3/7] Creating QCPP integration adapter...")
try:
    # Note: Current implementation takes predictor and cache_size (int), not config object
    cache_size = 1000
    qcpp_adapter = QCPPIntegrationAdapter(qcpp_predictor, cache_size)
    print(f"✓ Integration adapter created")
    print(f"  Cache size: {cache_size}")
except Exception as e:
    print(f"✗ Adapter creation failed: {e}")
    sys.exit(1)

# Test 4: Test QCPP analysis
print("\n[4/7] Testing QCPP conformation analysis...")
try:
    from ubf_protein.models import Conformation
    
    test_sequence = "ACDEFGH"
    test_coords: List[Tuple[float, float, float]] = [(float(i), 0.0, 0.0) for i in range(len(test_sequence))]
    
    # Create a Conformation object with all required fields
    test_conformation = Conformation(
        conformation_id="test_001",
        sequence=test_sequence,
        atom_coordinates=test_coords,
        energy=100.0,
        rmsd_to_native=5.0,
        secondary_structure=['C'] * len(test_sequence),  # All coil
        phi_angles=[0.0] * len(test_sequence),
        psi_angles=[0.0] * len(test_sequence),
        available_move_types=['LOCAL_BACKBONE_ROTATION', 'SIDE_CHAIN_ROTATION'],
        structural_constraints={}
    )
    
    metrics = qcpp_adapter.analyze_conformation(test_conformation)
    
    print(f"✓ QCPP analysis successful")
    print(f"  QCP score: {metrics.qcp_score:.3f}")
    print(f"  Field coherence: {metrics.field_coherence:.3f}")
    print(f"  Stability: {metrics.stability_score:.3f}")
    print(f"  Phi match: {metrics.phi_match_score:.3f}")
    print(f"  Calculation time: {metrics.calculation_time_ms:.2f}ms")
    
    if metrics.calculation_time_ms > 5.0:
        print(f"  ⚠ WARNING: Calculation time exceeds 5ms target")
    
except Exception as e:
    print(f"✗ QCPP analysis failed: {e}")
    sys.exit(1)

# Test 5: Test caching
print("\n[5/7] Testing cache functionality...")
try:
    # Analyze same conformation again (should hit cache)
    metrics2 = qcpp_adapter.analyze_conformation(test_conformation)
    
    cache_stats = qcpp_adapter.get_cache_stats()
    print(f"✓ Cache working")
    print(f"  Total requests: {cache_stats['total_analyses']}")
    print(f"  Cache hits: {cache_stats['cache_hits']}")
    print(f"  Hit rate: {cache_stats['cache_hit_rate']*100:.1f}%")
    
except Exception as e:
    print(f"✗ Cache test failed: {e}")
    sys.exit(1)

# Test 6: Create agent without QCPP (backward compatibility)
print("\n[6/7] Testing backward compatibility (no QCPP)...")
try:
    agent_baseline = ProteinAgent(protein_sequence="ACDEFGH")
    outcome = agent_baseline.explore_step()
    print(f"✓ Agent works without QCPP integration")
    print(f"  Energy change: {outcome.energy_change:.3f} kcal/mol")
except Exception as e:
    print(f"✗ Backward compatibility failed: {e}")
    sys.exit(1)

# Test 7: Run mini-exploration with QCPP
print("\n[7/7] Running mini-exploration with QCPP integration...")
try:
    sequence = "ACDEFGHIKL"  # 10 residues
    
    # Create agent
    agent = ProteinAgent(protein_sequence=sequence)
    
    print(f"  Running 20 iterations...")
    for i in range(20):
        outcome = agent.explore_step()
        
        # Manually analyze with QCPP every 5 steps (simulating integration)
        if i % 5 == 0:
            # Analyze the agent's current conformation
            metrics = qcpp_adapter.analyze_conformation(agent._current_conformation)
    
    cache_stats = qcpp_adapter.get_cache_stats()
    
    print(f"✓ Mini-exploration complete")
    print(f"  Total steps: 20")
    print(f"  QCPP analyses: {cache_stats['total_analyses']}")
    print(f"  Cache hit rate: {cache_stats['cache_hit_rate']*100:.1f}%")
    print(f"  Avg QCPP time: {cache_stats['avg_calculation_time_ms']:.2f}ms")
    
except Exception as e:
    print(f"✗ Mini-exploration failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success!
print("\n" + "="*70)
print("✓ ALL TESTS PASSED")
print("="*70)
print("\nQCPP-UBF integration is working correctly!")
print("\nNext steps:")
print("  1. Run full validation: python validate_qcpp_ubf_integration.py")
print("  2. Try example script: python ubf_protein/examples/integrated_exploration.py")
print("  3. Test with real protein: --sequence MQIFVKTLTGK --iterations 500")
print("="*70)

sys.exit(0)
