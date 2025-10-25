"""
Demo script showing the molecular mechanics energy function in action.

This demonstrates how to use the energy function with sample protein conformations.
"""

try:
    from ubf_protein.energy_function import (
        MolecularMechanicsEnergy,
        validate_energy_calculation
    )
    from ubf_protein.models import Conformation
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from ubf_protein.energy_function import (
        MolecularMechanicsEnergy,
        validate_energy_calculation
    )
    from ubf_protein.models import Conformation


def create_sample_conformation(name: str, coords, secondary_structure: str):
    """Helper to create sample conformation."""
    n = len(coords)
    return Conformation(
        conformation_id=name,
        sequence="A" * n,
        atom_coordinates=coords,
        energy=0.0,
        rmsd_to_native=None,
        secondary_structure=list(secondary_structure),
        phi_angles=[0.0] * n,
        psi_angles=[0.0] * n,
        available_move_types=[],
        structural_constraints={}
    )


def main():
    print("=" * 70)
    print("MOLECULAR MECHANICS ENERGY FUNCTION DEMO")
    print("=" * 70)
    print()
    
    # Initialize energy calculator
    calc = MolecularMechanicsEnergy(force_field="amber")
    print("✓ Initialized AMBER-like force field calculator")
    print()
    
    # Example 1: Ideal geometry (should have low bond/angle energy)
    print("-" * 70)
    print("Example 1: Ideal Geometry (near equilibrium bond lengths)")
    print("-" * 70)
    coords_ideal = [(0.0, 0.0, 0.0), (3.8, 0.0, 0.0), (7.6, 0.0, 0.0), (11.4, 0.0, 0.0)]
    conf_ideal = create_sample_conformation("ideal", coords_ideal, "CCCC")
    
    components_ideal = calc.calculate_with_components(conf_ideal)
    print(f"Total Energy: {components_ideal['total']:.2f} kcal/mol")
    print(f"  Bond:          {components_ideal['bond']:.2f} kcal/mol")
    print(f"  Angle:         {components_ideal['angle']:.2f} kcal/mol")
    print(f"  Dihedral:      {components_ideal['dihedral']:.2f} kcal/mol")
    print(f"  Van der Waals: {components_ideal['vdw']:.2f} kcal/mol")
    print(f"  Electrostatic: {components_ideal['electrostatic']:.2f} kcal/mol")
    print(f"  H-bonds:       {components_ideal['hbond']:.2f} kcal/mol")
    print()
    
    # Example 2: Compact helical structure (should have H-bonds)
    print("-" * 70)
    print("Example 2: Compact Helical Structure")
    print("-" * 70)
    coords_helix = [
        (0.0, 0.0, 0.0),
        (3.8, 0.0, 0.0),
        (6.0, 2.5, 0.0),
        (8.0, 4.0, 1.5),
        (9.5, 6.5, 2.0),
        (10.5, 8.5, 3.0),
    ]
    conf_helix = create_sample_conformation("helix", coords_helix, "HHHHHH")
    
    components_helix = calc.calculate_with_components(conf_helix)
    print(f"Total Energy: {components_helix['total']:.2f} kcal/mol")
    print(f"  Bond:          {components_helix['bond']:.2f} kcal/mol")
    print(f"  Angle:         {components_helix['angle']:.2f} kcal/mol")
    print(f"  Dihedral:      {components_helix['dihedral']:.2f} kcal/mol")
    print(f"  Van der Waals: {components_helix['vdw']:.2f} kcal/mol")
    print(f"  Electrostatic: {components_helix['electrostatic']:.2f} kcal/mol")
    print(f"  H-bonds:       {components_helix['hbond']:.2f} kcal/mol (helix H-bonds!)")
    print()
    
    # Example 3: Extended chain (unfolded-like)
    print("-" * 70)
    print("Example 3: Extended Chain (unfolded-like)")
    print("-" * 70)
    coords_extended = [(float(i * 4.5), 0.0, 0.0) for i in range(8)]
    conf_extended = create_sample_conformation("extended", coords_extended, "C" * 8)
    
    components_extended = calc.calculate_with_components(conf_extended)
    print(f"Total Energy: {components_extended['total']:.2f} kcal/mol")
    print(f"  Bond:          {components_extended['bond']:.2f} kcal/mol (stretched)")
    print(f"  Angle:         {components_extended['angle']:.2f} kcal/mol")
    print(f"  Dihedral:      {components_extended['dihedral']:.2f} kcal/mol")
    print(f"  Van der Waals: {components_extended['vdw']:.2f} kcal/mol")
    print(f"  Electrostatic: {components_extended['electrostatic']:.2f} kcal/mol")
    print(f"  H-bonds:       {components_extended['hbond']:.2f} kcal/mol")
    print()
    
    # Example 4: Validation report
    print("-" * 70)
    print("Example 4: Full Validation Report")
    print("-" * 70)
    metrics = validate_energy_calculation(conf_helix, calc)
    print(metrics.get_validation_report())
    print()
    
    # Summary comparison
    print("=" * 70)
    print("ENERGY COMPARISON SUMMARY")
    print("=" * 70)
    print(f"Ideal geometry:      {components_ideal['total']:8.2f} kcal/mol")
    print(f"Compact helix:       {components_helix['total']:8.2f} kcal/mol")
    print(f"Extended chain:      {components_extended['total']:8.2f} kcal/mol")
    print()
    print("Note: In a real protein, we'd expect:")
    print("  - Folded proteins: -200 to -50 kcal/mol")
    print("  - Unfolded proteins: > -50 kcal/mol or positive")
    print()
    print("These simplified examples show the energy function is working.")
    print("Integration with UBF system will use realistic protein coordinates.")
    print("=" * 70)


if __name__ == "__main__":
    main()
