"""
GGTH Predictor Configuration Manager v2.0
Handles loading and saving configuration, especially MT5 Files path
Updated for unified_predictor_v8.py
"""

import os
import json
from typing import Optional, Dict, Any


class ConfigManager:
    """Manages configuration for GGTH Predictor"""

    DEFAULT_CONFIG = {
        "mt5_files_path": "",
        "version": "2.0",
        "models_dir": "models",
        "use_kalman": True,
        "default_symbol": "USDJPY",
        "prediction_interval_minutes": 60,
        "default_models": ["lstm", "transformer", "lgbm"],
        "available_models": ["lstm", "gru", "transformer", "tcn", "lgbm"]
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config manager

        Args:
            config_path: Path to config.json file. If None, looks in script directory.
        """
        if config_path is None:
            # Look for config.json in the same directory as this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, "config.json")

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults (in case new fields were added)
                full_config = self.DEFAULT_CONFIG.copy()
                full_config.update(config)
                return full_config
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
                print("Using default configuration.")

        # Config doesn't exist or failed to load - return defaults
        return self.DEFAULT_CONFIG.copy()

    def save_config(self) -> bool:
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            return True
        except Exception as e:
            print(f"Error: Could not save config to {self.config_path}: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self.config[key] = value

    def get_mt5_files_path(self) -> str:
        """
        Get MT5 Files directory path

        Returns:
            Path to MT5 Files directory

        Raises:
            ValueError: If MT5 path is not configured
        """
        path = self.config.get("mt5_files_path", "")

        if not path:
            raise ValueError(
                "MT5 Files path is not configured!\n\n"
                "Please run the installer or manually create config.json with:\n"
                '{\n'
                '  "mt5_files_path": "C:\\\\Users\\\\YourName\\\\AppData\\\\Roaming\\\\MetaQuotes\\\\Terminal\\\\HASH\\\\MQL5\\\\Files"\n'
                '}'
            )

        if not os.path.exists(path):
            raise ValueError(
                f"MT5 Files path does not exist: {path}\n\n"
                "Please update config.json with the correct path."
            )

        return path

    def set_mt5_files_path(self, path: str) -> bool:
        """
        Set MT5 Files directory path

        Args:
            path: Path to MT5 Files directory

        Returns:
            True if path is valid and saved, False otherwise
        """
        if not os.path.exists(path):
            print(f"Error: Directory does not exist: {path}")
            return False

        self.config["mt5_files_path"] = path
        return self.save_config()

    def auto_detect_mt5_path(self) -> Optional[str]:
        """
        Attempt to auto-detect MT5 Files directory

        Returns:
            Path if found, None otherwise
        """
        import glob

        # Check common locations
        appdata = os.environ.get('APPDATA', '')
        if appdata:
            # Look for MetaQuotes Terminal folders
            terminal_base = os.path.join(appdata, 'MetaQuotes', 'Terminal')
            if os.path.exists(terminal_base):
                # Look for any hash folder with MQL5\Files
                pattern = os.path.join(terminal_base, '*', 'MQL5', 'Files')
                matches = glob.glob(pattern)
                if matches:
                    # Return the first match (most recent installation)
                    return matches[0]

        # Check Program Files
        program_files = os.environ.get('PROGRAMFILES', 'C:\\Program Files')
        mt5_path = os.path.join(program_files, 'MetaTrader 5', 'MQL5', 'Files')
        if os.path.exists(mt5_path):
            return mt5_path

        return None

    def get_default_models(self) -> list:
        """Get default model types for training"""
        return self.config.get("default_models", ["lstm", "transformer", "lgbm"])

    def get_available_models(self) -> list:
        """Get all available model types"""
        return self.config.get("available_models", ["lstm", "gru", "transformer", "tcn", "lgbm"])

    def print_config(self) -> None:
        """Print current configuration"""
        print("\n" + "=" * 50)
        print("  GGTH Predictor Configuration v2.0")
        print("=" * 50)
        print(f"Config file: {self.config_path}")
        print("\nSettings:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")
        print("=" * 50 + "\n")


# Global config instance
_config_instance = None


def get_config() -> ConfigManager:
    """Get global config instance (singleton pattern)"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance


def get_mt5_files_path() -> str:
    """Convenience function to get MT5 Files path"""
    return get_config().get_mt5_files_path()


if __name__ == "__main__":
    # Test/configure utility
    import argparse

    parser = argparse.ArgumentParser(description="GGTH Predictor Configuration Utility v2.0")
    parser.add_argument("--set-mt5-path", help="Set MT5 Files directory path")
    parser.add_argument("--auto-detect", action="store_true", help="Auto-detect MT5 path")
    parser.add_argument("--show", action="store_true", help="Show current configuration")
    parser.add_argument("--list-models", action="store_true", help="List available model types")

    args = parser.parse_args()
    config = get_config()

    if args.set_mt5_path:
        if config.set_mt5_files_path(args.set_mt5_path):
            print(f"✓ MT5 path set to: {args.set_mt5_path}")
        else:
            print("✗ Failed to set MT5 path")

    if args.auto_detect:
        path = config.auto_detect_mt5_path()
        if path:
            print(f"Found MT5 installation at: {path}")
            response = input("Use this path? (y/n): ")
            if response.lower() == 'y':
                if config.set_mt5_files_path(path):
                    print("✓ MT5 path configured successfully")
        else:
            print("Could not auto-detect MT5 installation")

    if args.list_models:
        print("\nAvailable model types:")
        for model in config.get_available_models():
            default = " (default)" if model in config.get_default_models() else ""
            print(f"  - {model}{default}")

    if args.show or (not args.set_mt5_path and not args.auto_detect and not args.list_models):
        config.print_config()
