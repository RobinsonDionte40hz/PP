"""
Configuration mapper to convert API parameters to PP system parameters.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigMapper:
    """Maps API configuration to PP system parameters."""
    
    # Preset configurations
    PRESETS = {
        "fast": {
            "iterations": 500,
            "agents": 5,
            "enable_qcpp": False,
            "qcpp_frequency": 10,
        },
        "balanced": {
            "iterations": 1000,
            "agents": 10,
            "enable_qcpp": True,
            "qcpp_frequency": 5,
        },
        "accurate": {
            "iterations": 2000,
            "agents": 20,
            "enable_qcpp": True,
            "qcpp_frequency": 1,
        },
        "high_performance": {
            "iterations": 5000,
            "agents": 50,
            "enable_qcpp": True,
            "qcpp_frequency": 5,
        },
    }
    
    @staticmethod
    def get_preset(preset_name: str) -> Dict[str, Any]:
        """
        Get a preset configuration.
        
        Args:
            preset_name: Name of the preset
            
        Returns:
            Preset configuration dictionary
        """
        if preset_name not in ConfigMapper.PRESETS:
            logger.warning(f"Unknown preset: {preset_name}, using 'balanced'")
            return ConfigMapper.PRESETS["balanced"].copy()
        
        logger.info(f"Retrieved preset: {preset_name}")
        return ConfigMapper.PRESETS[preset_name].copy()
    
    @staticmethod
    def map_api_to_pp_config(api_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map API configuration to PP system parameters.
        
        Args:
            api_config: Configuration from API request
            
        Returns:
            PP system configuration
        """
        # Start with preset if specified
        preset_name = api_config.get("preset", "balanced")
        pp_config = ConfigMapper.get_preset(preset_name)
        
        # Override with custom values
        if "iterations" in api_config:
            pp_config["iterations"] = api_config["iterations"]
        
        if "agents" in api_config:
            pp_config["agents"] = api_config["agents"]
        
        if "enable_qcpp" in api_config:
            pp_config["enable_qcpp"] = api_config["enable_qcpp"]
        
        if "qcpp_frequency" in api_config:
            pp_config["qcpp_frequency"] = api_config["qcpp_frequency"]
        
        # Map exploration parameters
        if "exploration_params" in api_config:
            ep = api_config["exploration_params"]
            if "aggressiveness" in ep:
                pp_config["aggressiveness"] = ep["aggressiveness"]
            if "consistency" in ep:
                pp_config["consistency"] = ep["consistency"]
        
        # Map diversity settings
        if "diversity" in api_config:
            pp_config["diversity"] = api_config["diversity"]
        
        # Map checkpoint settings
        if "checkpoint_interval" in api_config:
            pp_config["checkpoint_interval"] = api_config["checkpoint_interval"]
        
        logger.info(f"Mapped API config to PP config: {pp_config}")
        return pp_config
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate iterations
        iterations = config.get("iterations", 0)
        if iterations < 1 or iterations > 10000:
            return False, "Iterations must be between 1 and 10000"
        
        # Validate agents
        agents = config.get("agents", 0)
        if agents < 1 or agents > 100:
            return False, "Agents must be between 1 and 100"
        
        # Validate QCPP frequency
        if "qcpp_frequency" in config:
            freq = config["qcpp_frequency"]
            if freq < 1 or freq > 100:
                return False, "QCPP frequency must be between 1 and 100"
        
        # Validate exploration parameters
        if "aggressiveness" in config:
            agg = config["aggressiveness"]
            if agg < 3.0 or agg > 15.0:
                return False, "Aggressiveness must be between 3.0 and 15.0"
        
        if "consistency" in config:
            cons = config["consistency"]
            if cons < 0.2 or cons > 1.0:
                return False, "Consistency must be between 0.2 and 1.0"
        
        logger.info("Configuration validation passed")
        return True, None
