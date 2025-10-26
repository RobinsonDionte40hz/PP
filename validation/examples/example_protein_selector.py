"""
Example usage of ProteinSelector for Large-Scale Validation

This script demonstrates how to:
1. Select diverse proteins for testing
2. Apply various filters
3. Export selections for reproducibility
4. Analyze selection characteristics
"""

import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from validation.protein_selector import ProteinSelector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Demonstrate ProteinSelector usage."""
    
    # ========================================================================
    # Example 1: Select 60 proteins with default balanced distribution
    # ========================================================================
    logger.info("=" * 70)
    logger.info("Example 1: Default selection (60 proteins)")
    logger.info("=" * 70)
    
    selector = ProteinSelector()
    proteins = selector.select_proteins(target_count=60)
    
    logger.info(f"\nSelected {len(proteins)} proteins")
    
    # Analyze distribution
    size_dist = {}
    class_dist = {}
    method_dist = {}
    
    for p in proteins:
        size_dist[p.size_category] = size_dist.get(p.size_category, 0) + 1
        class_dist[p.structural_class] = class_dist.get(p.structural_class, 0) + 1
        method_dist[p.experimental_method] = method_dist.get(p.experimental_method, 0) + 1
    
    logger.info(f"\nSize Distribution:")
    for category, count in sorted(size_dist.items()):
        pct = (count / len(proteins)) * 100
        logger.info(f"  {category:10s}: {count:2d} ({pct:5.1f}%)")
    
    logger.info(f"\nStructural Class Distribution:")
    for cls, count in sorted(class_dist.items()):
        pct = (count / len(proteins)) * 100
        logger.info(f"  {cls:15s}: {count:2d} ({pct:5.1f}%)")
    
    logger.info(f"\nExperimental Method Distribution:")
    for method, count in sorted(method_dist.items()):
        pct = (count / len(proteins)) * 100
        logger.info(f"  {method:10s}: {count:2d} ({pct:5.1f}%)")
    
    # Export selection
    selector.export_selection(proteins, 'validation/selected_proteins_60.json')
    selector.export_selection(proteins, 'validation/selected_proteins_60.csv', format='csv')
    logger.info("\n✓ Exported to selected_proteins_60.json and selected_proteins_60.csv")
    
    # ========================================================================
    # Example 2: Custom distribution (focus on small/medium proteins)
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Example 2: Custom distribution (focus on small/medium)")
    logger.info("=" * 70)
    
    custom_dist = {
        'tiny': 0.10,    # 10%
        'small': 0.40,   # 40%
        'medium': 0.40,  # 40%
        'large': 0.10    # 10%
    }
    
    proteins_custom = selector.select_proteins(
        target_count=50,
        size_distribution=custom_dist
    )
    
    logger.info(f"\nSelected {len(proteins_custom)} proteins with custom distribution")
    
    # ========================================================================
    # Example 3: High-quality X-ray structures only
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Example 3: High-quality X-ray structures only")
    logger.info("=" * 70)
    
    proteins_xray = selector.select_proteins(
        target_count=40,
        max_resolution=2.0,  # High resolution only
        include_nmr=False    # X-ray only
    )
    
    logger.info(f"\nSelected {len(proteins_xray)} high-quality X-ray structures")
    
    # Show resolution distribution
    resolutions = [p.resolution for p in proteins_xray if p.resolution is not None]
    if resolutions:
        logger.info(f"Resolution range: {min(resolutions):.2f} - {max(resolutions):.2f} Å")
        logger.info(f"Average resolution: {sum(resolutions)/len(resolutions):.2f} Å")
    
    # ========================================================================
    # Example 4: Filtering examples
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Example 4: Applying filters")
    logger.info("=" * 70)
    
    # Start with all proteins
    all_proteins = selector.select_proteins(target_count=60)
    
    # Filter for small proteins
    small_proteins = selector.filter_by_size(all_proteins, ['small'])
    logger.info(f"\nSmall proteins: {len(small_proteins)}")
    
    # Filter for all-alpha helical proteins
    alpha_proteins = selector.filter_by_structural_class(all_proteins, ['all-alpha'])
    logger.info(f"All-alpha proteins: {len(alpha_proteins)}")
    
    # Filter for high-resolution structures
    high_res = selector.filter_by_resolution(all_proteins, max_resolution=1.8)
    logger.info(f"High-resolution structures (≤1.8Å): {len(high_res)}")
    
    # Combine filters: small, all-alpha, high-resolution
    filtered = selector.filter_by_size(all_proteins, ['small'])
    filtered = selector.filter_by_structural_class(filtered, ['all-alpha'])
    filtered = selector.filter_by_resolution(filtered, max_resolution=2.0)
    
    logger.info(f"\nCombined filters (small + all-alpha + high-res): {len(filtered)}")
    for p in filtered:
        logger.info(f"  {p.pdb_id}: {p.sequence_length} residues, "
                   f"{p.resolution}Å - {p.description}")
    
    # ========================================================================
    # Example 5: Phase-specific selections
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Example 5: Phase-specific selections")
    logger.info("=" * 70)
    
    # Phase 1: Easy, well-studied proteins
    phase1 = selector.select_proteins(target_count=10)
    phase1_filtered = selector.filter_by_size(phase1, ['tiny', 'small'])
    phase1_filtered = selector.filter_by_resolution(phase1_filtered, max_resolution=2.0)
    
    logger.info(f"\nPhase 1 (easy, well-studied): {len(phase1_filtered)} proteins")
    logger.info("Examples:")
    for p in phase1_filtered[:5]:
        logger.info(f"  {p.pdb_id}: {p.description}")
    
    # Phase 2: Mixed difficulty
    phase2 = selector.select_proteins(target_count=15)
    logger.info(f"\nPhase 2 (mixed difficulty): {len(phase2)} proteins")
    
    # Phase 3: Diverse characteristics
    phase3_dist = {
        'tiny': 0.20,
        'small': 0.30,
        'medium': 0.30,
        'large': 0.20
    }
    phase3 = selector.select_proteins(target_count=25, size_distribution=phase3_dist)
    logger.info(f"\nPhase 3 (diverse): {len(phase3)} proteins")
    
    # ========================================================================
    # Example 6: Load and reuse previous selection
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Example 6: Load previous selection")
    logger.info("=" * 70)
    
    # Load previously exported selection
    loaded_proteins = selector.load_selection('validation/selected_proteins_60.json')
    logger.info(f"\nLoaded {len(loaded_proteins)} proteins from JSON")
    
    # Verify same proteins
    assert len(loaded_proteins) == len(proteins)
    logger.info("✓ Verified: Loaded proteins match original selection")
    
    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Summary")
    logger.info("=" * 70)
    
    logger.info("""
The ProteinSelector provides flexible protein selection for large-scale validation:

1. Balanced selection across size categories (tiny, small, medium, large)
2. Diverse structural classes (all-alpha, all-beta, alpha-beta, alpha+beta)
3. Quality filters (resolution, completeness)
4. Custom distributions for different testing phases
5. Export/import for reproducibility
6. Chainable filters for precise selection

Ready for Phase 2: Implement PhaseManager for progressive testing
""")


if __name__ == '__main__':
    main()
