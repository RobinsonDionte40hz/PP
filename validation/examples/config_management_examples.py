"""
Configuration Management Examples

This script demonstrates comprehensive usage of the configuration management system
for large-scale protein validation campaigns.

Key Features Demonstrated:
1. Loading presets
2. Loading from files
3. Overriding configurations
4. Validating configurations
5. Saving custom configurations
6. Merging configurations
7. Creating preset files
8. Error handling

For complete documentation, see: validation/configs/README.md
"""

import logging
from pathlib import Path
from validation.campaign_config import (
    CampaignConfigManager,
    ConfigPresets,
    ConfigValidator,
    ConfigLoader,
    ValidationError,
    CampaignPreset
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Example 1: Loading Presets
# ============================================================================

def example_1_loading_presets():
    """
    Demonstrate loading different configuration presets.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Loading Presets")
    print("="*80)
    
    manager = CampaignConfigManager()
    
    # List all available presets
    print("\nAvailable presets:")
    for preset_name in manager.list_presets():
        print(f"  - {preset_name}")
    
    # Load each preset and display key settings
    for preset_name in ['default', 'fast', 'high_accuracy']:
        config = manager.get_preset(preset_name)
        print(f"\n{preset_name.upper()} Preset:")
        print(f"  Proteins: {config.target_protein_count}")
        print(f"  QCPP: {config.enable_qcpp}")
        print(f"  Agents: {config.num_agents}")
        print(f"  Iterations: {config.iterations_per_agent}")
        print(f"  Parallel: {config.max_parallel_tests}")


# ============================================================================
# Example 2: Loading from Files
# ============================================================================

def example_2_loading_from_files():
    """
    Demonstrate loading configurations from JSON files.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Loading from Files")
    print("="*80)
    
    manager = CampaignConfigManager()
    
    # Load from JSON file
    config_path = Path("./validation/configs/default_campaign.json")
    if config_path.exists():
        config = manager.load(config_path)
        print(f"\nLoaded from {config_path}:")
        print(f"  Proteins: {config.target_protein_count}")
        print(f"  Agents: {config.num_agents}")
        print(f"  Output: {config.output_dir}")
    else:
        print(f"\nConfig file not found: {config_path}")
        print("Creating example config...")
        
        # Create example config
        config = manager.get_preset('default')
        manager.save(config, config_path)
        print(f"✓ Created {config_path}")


# ============================================================================
# Example 3: Overriding Configurations
# ============================================================================

def example_3_overriding_configurations():
    """
    Demonstrate overriding configuration parameters.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Overriding Configurations")
    print("="*80)
    
    manager = CampaignConfigManager()
    
    # Start with default preset
    base_config = manager.get_preset('default')
    print(f"\nBase configuration (default):")
    print(f"  Agents: {base_config.num_agents}")
    print(f"  Iterations: {base_config.iterations_per_agent}")
    print(f"  Seed: {base_config.random_seed}")
    
    # Override specific parameters
    custom_config = manager.override(
        base_config,
        num_agents=15,
        iterations_per_agent=1500,
        random_seed=123,
        output_dir="./custom_results"
    )
    
    print(f"\nCustom configuration (overridden):")
    print(f"  Agents: {custom_config.num_agents}")
    print(f"  Iterations: {custom_config.iterations_per_agent}")
    print(f"  Seed: {custom_config.random_seed}")
    print(f"  Output: {custom_config.output_dir}")


# ============================================================================
# Example 4: Validating Configurations
# ============================================================================

def example_4_validating_configurations():
    """
    Demonstrate configuration validation.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Validating Configurations")
    print("="*80)
    
    manager = CampaignConfigManager()
    
    # Valid configuration
    print("\n1. Validating a valid configuration:")
    valid_config = manager.get_preset('default')
    try:
        manager.validate(valid_config)
        print("✓ Configuration is valid")
    except ValidationError as e:
        print(f"✗ Validation failed: {e}")
    
    # Invalid configuration (negative agents)
    print("\n2. Validating an invalid configuration:")
    invalid_config = manager.override(valid_config, num_agents=-5)
    try:
        manager.validate(invalid_config)
        print("✓ Configuration is valid")
    except ValidationError as e:
        print(f"✗ Validation failed (expected):")
        print(f"  {e}")
    
    # Configuration with warnings
    print("\n3. Validating configuration with warnings:")
    warning_config = manager.override(
        valid_config,
        target_protein_count=15,  # Very small
        num_agents=30             # High
    )
    try:
        manager.validate(warning_config)
        print("✓ Configuration is valid (check logs for warnings)")
    except ValidationError as e:
        print(f"✗ Validation failed: {e}")


# ============================================================================
# Example 5: Saving Custom Configurations
# ============================================================================

def example_5_saving_configurations():
    """
    Demonstrate saving custom configurations to files.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Saving Custom Configurations")
    print("="*80)
    
    manager = CampaignConfigManager()
    
    # Create custom configuration
    custom_config = manager.override(
        manager.get_preset('default'),
        target_protein_count=50,
        num_agents=12,
        iterations_per_agent=1200,
        random_seed=42,
        output_dir="./my_experiment_results"
    )
    
    # Save to JSON
    json_path = Path("./validation/configs/my_custom_campaign.json")
    manager.save(custom_config, json_path)
    print(f"\n✓ Saved to {json_path}")
    
    # Try YAML (if PyYAML installed)
    try:
        yaml_path = Path("./validation/configs/my_custom_campaign.yaml")
        manager.save(custom_config, yaml_path, format='yaml')
        print(f"✓ Saved to {yaml_path}")
    except ImportError:
        print("✗ YAML export requires PyYAML: pip install pyyaml")


# ============================================================================
# Example 6: Merging Configurations
# ============================================================================

def example_6_merging_configurations():
    """
    Demonstrate merging multiple configuration sources.
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Merging Configurations")
    print("="*80)
    
    manager = CampaignConfigManager()
    
    # Start with a preset
    print("\n1. Load base preset (high_accuracy):")
    base_config = manager.get_preset('high_accuracy')
    print(f"  Agents: {base_config.num_agents}")
    print(f"  Iterations: {base_config.iterations_per_agent}")
    print(f"  Parallel: {base_config.max_parallel_tests}")
    
    # Override with file (if exists)
    override_file = Path("./validation/configs/custom_overrides.json")
    if override_file.exists():
        print(f"\n2. Apply overrides from {override_file}:")
        final_config = manager.load_with_preset('high_accuracy', override_file)
        print(f"  Agents: {final_config.num_agents}")
        print(f"  Iterations: {final_config.iterations_per_agent}")
        print(f"  Parallel: {final_config.max_parallel_tests}")
    else:
        print(f"\n2. No override file found at {override_file}")
        
        # Create example override file
        overrides = {
            "num_agents": 15,
            "max_parallel_tests": 4
        }
        import json
        override_file.parent.mkdir(parents=True, exist_ok=True)
        with open(override_file, 'w') as f:
            json.dump(overrides, f, indent=2)
        print(f"   Created example override file")
    
    # Further override with direct parameters
    print("\n3. Apply additional direct overrides:")
    final_config = manager.override(
        base_config,
        random_seed=999,
        output_dir="./final_results"
    )
    print(f"  Seed: {final_config.random_seed}")
    print(f"  Output: {final_config.output_dir}")


# ============================================================================
# Example 7: Creating All Preset Files
# ============================================================================

def example_7_creating_preset_files():
    """
    Demonstrate creating all preset configuration files.
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: Creating All Preset Files")
    print("="*80)
    
    manager = CampaignConfigManager()
    
    output_dir = Path("./validation/configs")
    print(f"\nCreating preset files in {output_dir}...")
    
    manager.create_preset_files(output_dir)
    
    print(f"\n✓ Created preset files:")
    for preset_name in manager.list_presets():
        file_path = output_dir / f"{preset_name}_campaign.json"
        if file_path.exists():
            print(f"  - {file_path}")


# ============================================================================
# Example 8: Using Presets with ValidationSuite
# ============================================================================

def example_8_using_with_campaign():
    """
    Demonstrate using configuration management with actual campaign.
    """
    print("\n" + "="*80)
    print("EXAMPLE 8: Using Configuration with Campaign")
    print("="*80)
    
    from validation.large_scale_validation_campaign import LargeScaleValidationCampaign
    
    manager = CampaignConfigManager()
    
    # Load and customize configuration
    config = manager.get_preset('development')  # Fast preset for demo
    config = manager.override(
        config,
        target_protein_count=5,  # Very small for demo
        output_dir="./demo_campaign_results"
    )
    
    # Validate before use
    print("\nValidating configuration...")
    manager.validate(config)
    print("✓ Configuration valid")
    
    # Create campaign with validated config
    print("\nCreating campaign with configuration...")
    campaign = LargeScaleValidationCampaign(config=config)
    print(f"✓ Campaign created with ID: {campaign.campaign_id}")
    print(f"  Output directory: {campaign.output_dir}")
    
    # Note: Don't actually run the campaign in this example
    print("\nNote: Campaign not executed (example only)")


# ============================================================================
# Example 9: Error Handling
# ============================================================================

def example_9_error_handling():
    """
    Demonstrate proper error handling with configuration management.
    """
    print("\n" + "="*80)
    print("EXAMPLE 9: Error Handling")
    print("="*80)
    
    manager = CampaignConfigManager()
    
    # 1. Invalid preset name
    print("\n1. Handling invalid preset name:")
    try:
        config = manager.get_preset('nonexistent_preset')
        print("✓ Loaded preset")
    except ValueError as e:
        print(f"✗ Error (expected): {e}")
    
    # 2. File not found
    print("\n2. Handling missing configuration file:")
    try:
        config = manager.load('./nonexistent_config.json')
        print("✓ Loaded configuration")
    except FileNotFoundError as e:
        print(f"✗ Error (expected): {e}")
    
    # 3. Validation failure
    print("\n3. Handling validation errors:")
    try:
        invalid_config = manager.override(
            manager.get_preset('default'),
            num_agents=0,  # Invalid
            quality_gate_threshold=2.0  # Invalid
        )
        manager.validate(invalid_config)
        print("✓ Configuration valid")
    except ValidationError as e:
        print(f"✗ Validation error (expected):\n{e}")
    
    # 4. Non-strict validation
    print("\n4. Using non-strict validation:")
    invalid_config = manager.override(
        manager.get_preset('default'),
        target_protein_count=5  # Will generate warning
    )
    is_valid = manager.validate(invalid_config, strict=False)
    print(f"  Valid: {is_valid} (check logs for warnings)")


# ============================================================================
# Example 10: Complete Workflow
# ============================================================================

def example_10_complete_workflow():
    """
    Demonstrate complete configuration workflow for research project.
    """
    print("\n" + "="*80)
    print("EXAMPLE 10: Complete Research Workflow")
    print("="*80)
    
    manager = CampaignConfigManager()
    
    # Step 1: Start with appropriate preset
    print("\n1. Select base configuration for research goal:")
    print("   Goal: Publication-quality validation with modifications")
    base_config = manager.get_preset('high_accuracy')
    print(f"   ✓ Loaded 'high_accuracy' preset")
    
    # Step 2: Customize for specific research needs
    print("\n2. Customize for specific research needs:")
    custom_config = manager.override(
        base_config,
        target_protein_count=60,  # Reduce from 75 for time
        num_agents=15,            # Reduce from 20
        output_dir="./my_publication_results",
        random_seed=2025          # Current year for reproducibility
    )
    print(f"   ✓ Customized configuration")
    
    # Step 3: Validate
    print("\n3. Validate configuration:")
    try:
        manager.validate(custom_config)
        print("   ✓ Configuration valid")
    except ValidationError as e:
        print(f"   ✗ Validation failed: {e}")
        return
    
    # Step 4: Save for reproducibility
    print("\n4. Save configuration for reproducibility:")
    config_path = Path("./validation/configs/my_publication_2025.json")
    manager.save(custom_config, config_path)
    print(f"   ✓ Saved to {config_path}")
    
    # Step 5: Document configuration
    print("\n5. Configuration summary:")
    print(f"   - Proteins: {custom_config.target_protein_count}")
    print(f"   - QCPP: {'Enabled' if custom_config.enable_qcpp else 'Disabled'}")
    print(f"   - Agents: {custom_config.num_agents}")
    print(f"   - Iterations: {custom_config.iterations_per_agent}")
    print(f"   - Seed: {custom_config.random_seed}")
    print(f"   - Output: {custom_config.output_dir}")
    
    print("\n6. Ready to execute campaign:")
    print(f"   python validation/run_validation_campaign.py \\")
    print(f"       --config {config_path} \\")
    print(f"       --batch \\")
    print(f"       --log-file ./publication_campaign.log")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("CONFIGURATION MANAGEMENT EXAMPLES")
    print("="*80)
    print("\nThese examples demonstrate the configuration management system")
    print("for large-scale protein validation campaigns.")
    print("\nNote: Some examples create files in ./validation/configs/")
    
    # Run all examples
    try:
        example_1_loading_presets()
        example_2_loading_from_files()
        example_3_overriding_configurations()
        example_4_validating_configurations()
        example_5_saving_configurations()
        example_6_merging_configurations()
        example_7_creating_preset_files()
        example_8_using_with_campaign()
        example_9_error_handling()
        example_10_complete_workflow()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED")
        print("="*80)
        print("\nFor more information, see:")
        print("  - validation/configs/README.md")
        print("  - validation/campaign_config.py")
        print("  - validation/run_validation_campaign.py")
        
    except Exception as e:
        logger.error(f"Example failed with error: {e}", exc_info=True)
