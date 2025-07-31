"""
Development utilities configuration system.

This module provides centralized configuration for all development utilities
including debugging, logging, profiling, and static analysis tools.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _parse_environment_from_env() -> "Environment":
    """Parse environment from HAIVE_ENV variable, handling file paths."""
    env_var = os.getenv("HAIVE_ENV", "development")
    if env_var.endswith(".env") or "/" in env_var:
        # HAIVE_ENV is set to a file path, use default
        env_name = "development"
    else:
        env_name = env_var
    return Environment(env_name)


class Environment(str, Enum):
    """Supported runtime environments.

    Attributes:
        DEVELOPMENT: Local development environment with full debugging
        TESTING: Test environment with reduced overhead
        STAGING: Staging environment with minimal debugging
        PRODUCTION: Production environment with error logging only
    """

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels for development utilities.

    Attributes:
        TRACE: Most verbose logging including execution traces
        DEBUG: Debug information for development
        INFO: General information messages
        WARNING: Warning messages for potential issues
        ERROR: Error messages only
    """

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class StorageBackend(str, Enum):
    """Storage backends for analysis data.

    Attributes:
        NONE: No persistent storage
        MEMORY: In-memory storage (lost on restart)
        SQLITE: SQLite database storage
        POSTGRESQL: PostgreSQL database storage
        FILE: File-based storage
    """

    NONE = "none"
    MEMORY = "memory"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    FILE = "file"


@dataclass
class DevConfig:
    """Centralized configuration for all development utilities.

    This class provides comprehensive configuration options for debugging,
    logging, profiling, tracing, and static analysis tools. It automatically
    adjusts settings based on the runtime environment.

    Attributes:
        enabled: Whether development utilities are enabled globally
        environment: Current runtime environment
        debug_enabled: Enable debug utilities (icecream, pdb, etc.)
        log_enabled: Enable logging utilities
        trace_enabled: Enable execution tracing
        profile_enabled: Enable performance profiling
        benchmark_enabled: Enable benchmarking utilities
        static_analysis_enabled: Enable static analysis tools
        production_safe: Automatically reduce overhead in production
        trace_sampling_rate: Percentage of traces to capture (0.0-1.0)
        profile_overhead_limit: Maximum acceptable performance overhead
        storage_enabled: Enable persistent storage of analysis data
        storage_backend: Backend for storing analysis data
        storage_path: Path for file-based storage
        storage_config: Additional storage configuration
        use_rich: Enable rich terminal formatting
        color_enabled: Enable colored output
        verbose: Enable verbose output
        log_level: Minimum logging level
        auto_analyze: Automatically analyze instrumented functions
        runtime_type_check: Enable runtime type checking
        correlation_enabled: Enable correlation ID tracking
        dashboard_enabled: Enable web dashboard
        dashboard_port: Port for web dashboard
        max_file_size: Maximum size for analysis files (bytes)
        retention_days: Days to retain analysis data
        excluded_paths: Paths to exclude from analysis
        included_tools: Static analysis tools to include
        excluded_tools: Static analysis tools to exclude

    Examples:
        Basic usage with defaults::

            config = DevConfig()
            print(f"Environment: {config.environment}")
            print(f"Debug enabled: {config.debug_enabled}")

        Custom configuration::

            config = DevConfig(
                environment=Environment.PRODUCTION,
                trace_sampling_rate=0.01,
                storage_backend=StorageBackend.POSTGRESQL
            )

        Environment-based configuration::

            config = DevConfig.from_env()
            config.configure_for_testing()
    """

    # Global settings
    enabled: bool = True
    environment: Environment = field(
        default_factory=lambda: _parse_environment_from_env()
    )

    # Component toggles
    debug_enabled: bool = True
    log_enabled: bool = True
    trace_enabled: bool = True
    profile_enabled: bool = True
    benchmark_enabled: bool = True
    static_analysis_enabled: bool = True

    # Performance settings
    production_safe: bool = field(
        default_factory=lambda: os.getenv("HAIVE_ENV") == "production"
    )
    trace_sampling_rate: float = 1.0
    profile_overhead_limit: float = 0.05

    # Storage settings
    storage_enabled: bool = True
    storage_backend: StorageBackend = StorageBackend.SQLITE
    storage_path: str = ".haive_dev_data"
    storage_config: Dict[str, Any] = field(default_factory=dict)

    # Output settings
    use_rich: bool = True
    color_enabled: bool = True
    verbose: bool = field(
        default_factory=lambda: os.getenv("HAIVE_DEBUG", "false").lower() == "true"
    )
    log_level: LogLevel = LogLevel.INFO

    # Advanced features
    auto_analyze: bool = False
    runtime_type_check: bool = False
    correlation_enabled: bool = True

    # Dashboard settings
    dashboard_enabled: bool = False
    dashboard_port: int = 8888

    # Data management
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    retention_days: int = 30

    # Internal field to track environment variable overrides
    _env_overrides: Dict[str, Any] = field(default_factory=dict, init=True, repr=False)

    # Tool selection
    excluded_paths: List[str] = field(
        default_factory=lambda: [
            "**/test_*",
            "**/*_test.py",
            "**/__pycache__/**",
            "**/.*",
        ]
    )
    included_tools: Optional[List[str]] = None
    excluded_tools: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Configure settings based on environment after initialization.

        Automatically adjusts configuration for production safety and
        performance optimization based on the detected environment.
        """
        if self.production_safe or self.environment == Environment.PRODUCTION:
            self._configure_for_production()
        elif self.environment == Environment.TESTING:
            self._configure_for_testing()
        elif self.environment == Environment.STAGING:
            self._configure_for_staging()

    def _configure_for_production(self) -> None:
        """Configure for production environment with minimal overhead."""
        self.trace_sampling_rate = 0.01  # 1% sampling
        self.profile_enabled = False
        self.benchmark_enabled = False
        self.static_analysis_enabled = False
        self.verbose = False
        self.log_level = LogLevel.ERROR
        self.storage_backend = StorageBackend.NONE
        self.dashboard_enabled = False
        self.auto_analyze = False
        self.runtime_type_check = False

    def _configure_for_testing(self) -> None:
        """Configure for testing environment with balanced features."""
        # Debug print
        print(f"DEBUG: _env_overrides = {self._env_overrides}")
        print(
            f"DEBUG: trace_sampling_rate in overrides? {'trace_sampling_rate' in self._env_overrides}"
        )

        if "trace_sampling_rate" not in self._env_overrides:
            self.trace_sampling_rate = 0.1  # 10% sampling
        if "profile_enabled" not in self._env_overrides:
            self.profile_enabled = True
        if "benchmark_enabled" not in self._env_overrides:
            self.benchmark_enabled = False
        if "static_analysis_enabled" not in self._env_overrides:
            self.static_analysis_enabled = False
        if "verbose" not in self._env_overrides:
            self.verbose = False
        # Don't override log_level if it was set via environment variable
        if os.getenv("HAIVE_LOG_LEVEL") is None:
            self.log_level = LogLevel.WARNING
        self.storage_backend = StorageBackend.MEMORY
        self.dashboard_enabled = False

    def _configure_for_staging(self) -> None:
        """Configure for staging environment with monitoring focus."""
        self.trace_sampling_rate = 0.05  # 5% sampling
        self.profile_enabled = False
        self.benchmark_enabled = False
        self.static_analysis_enabled = False
        self.verbose = False
        self.log_level = LogLevel.INFO
        self.storage_backend = StorageBackend.SQLITE
        self.dashboard_enabled = True

    @classmethod
    def from_env(cls) -> "DevConfig":
        """Create configuration from environment variables.

        Reads configuration from environment variables with HAIVE_ prefix.
        Provides a convenient way to configure the system through environment
        variables in containerized or CI/CD environments.

        Returns:
            DevConfig: Configuration instance populated from environment

        Environment Variables:
            HAIVE_ENV: Runtime environment (development/testing/staging/production)
            HAIVE_DEV_ENABLED: Enable development utilities (true/false)
            HAIVE_DEBUG_ENABLED: Enable debug utilities (true/false)
            HAIVE_LOG_ENABLED: Enable logging utilities (true/false)
            HAIVE_TRACE_ENABLED: Enable tracing utilities (true/false)
            HAIVE_PROFILE_ENABLED: Enable profiling utilities (true/false)
            HAIVE_BENCHMARK_ENABLED: Enable benchmarking utilities (true/false)
            HAIVE_STATIC_ANALYSIS_ENABLED: Enable static analysis (true/false)
            HAIVE_TRACE_SAMPLING_RATE: Trace sampling rate (0.0-1.0)
            HAIVE_STORAGE_BACKEND: Storage backend (none/memory/sqlite/postgresql/file)
            HAIVE_STORAGE_PATH: Path for file-based storage
            HAIVE_LOG_LEVEL: Minimum logging level (TRACE/DEBUG/INFO/WARNING/ERROR)
            HAIVE_DASHBOARD_ENABLED: Enable web dashboard (true/false)
            HAIVE_DASHBOARD_PORT: Port for web dashboard (integer)
            HAIVE_VERBOSE: Enable verbose output (true/false)

        Examples:
            Load from environment::

                # Set environment variables
                os.environ["HAIVE_ENV"] = "production"
                os.environ["HAIVE_TRACE_SAMPLING_RATE"] = "0.01"

                # Create config
                config = DevConfig.from_env()
                assert config.environment == Environment.PRODUCTION
                assert config.trace_sampling_rate == 0.01
        """
        # Get environment, handling case where HAIVE_ENV might be a file path
        env_var = os.getenv("HAIVE_ENV", "development")
        if env_var.endswith(".env") or "/" in env_var:
            # HAIVE_ENV is set to a file path, use default
            env_name = "development"
        else:
            env_name = env_var

        # Track which environment variables were explicitly set
        overrides = {}

        # Helper to track overrides
        def _track_env(env_var: str, field_name: str, parser, default):
            value = parser(env_var, default)
            if os.getenv(env_var) is not None:
                overrides[field_name] = value
            return value

        config = cls(
            enabled=_track_env("HAIVE_DEV_ENABLED", "enabled", _env_bool, True),
            environment=Environment(env_name),
            debug_enabled=_track_env(
                "HAIVE_DEBUG_ENABLED", "debug_enabled", _env_bool, True
            ),
            log_enabled=_track_env("HAIVE_LOG_ENABLED", "log_enabled", _env_bool, True),
            trace_enabled=_track_env(
                "HAIVE_TRACE_ENABLED", "trace_enabled", _env_bool, True
            ),
            profile_enabled=_track_env(
                "HAIVE_PROFILE_ENABLED", "profile_enabled", _env_bool, True
            ),
            benchmark_enabled=_track_env(
                "HAIVE_BENCHMARK_ENABLED", "benchmark_enabled", _env_bool, True
            ),
            static_analysis_enabled=_track_env(
                "HAIVE_STATIC_ANALYSIS_ENABLED",
                "static_analysis_enabled",
                _env_bool,
                True,
            ),
            trace_sampling_rate=_track_env(
                "HAIVE_TRACE_SAMPLING_RATE", "trace_sampling_rate", _env_float, 1.0
            ),
            storage_backend=StorageBackend(
                os.getenv("HAIVE_STORAGE_BACKEND", "sqlite")
            ),
            storage_path=os.getenv("HAIVE_STORAGE_PATH", ".haive_dev_data"),
            log_level=LogLevel(os.getenv("HAIVE_LOG_LEVEL", "INFO")),
            dashboard_enabled=_track_env(
                "HAIVE_DASHBOARD_ENABLED", "dashboard_enabled", _env_bool, False
            ),
            dashboard_port=_track_env(
                "HAIVE_DASHBOARD_PORT", "dashboard_port", _env_int, 8888
            ),
            verbose=_track_env("HAIVE_VERBOSE", "verbose", _env_bool, False),
        )

        # Store overrides for later use in __post_init__
        config._env_overrides = overrides
        return config

    def update(self, **kwargs: Any) -> None:
        """Update configuration with new values.

        Args:
            **kwargs: Configuration values to update

        Examples:
            Update multiple values::

                config.update(
                    verbose=True,
                    log_level=LogLevel.DEBUG,
                    trace_sampling_rate=0.5
                )
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if a specific tool is enabled.

        Args:
            tool_name: Name of the tool to check

        Returns:
            bool: True if the tool is enabled, False otherwise

        Examples:
            Check tool availability::

                if config.is_tool_enabled("mypy"):
                    # Run mypy analysis
                    pass
        """
        if not self.static_analysis_enabled:
            return False

        if self.included_tools is not None:
            return tool_name in self.included_tools

        return tool_name not in self.excluded_tools

    def get_storage_config(self) -> Dict[str, Any]:
        """Get complete storage configuration.

        Returns:
            Dict[str, Any]: Storage configuration dictionary

        Examples:
            Get storage settings::

                storage_config = config.get_storage_config()
                backend = storage_config["backend"]
                path = storage_config["path"]
        """
        base_config = {
            "backend": self.storage_backend.value,
            "enabled": self.storage_enabled,
            "path": self.storage_path,
            "max_file_size": self.max_file_size,
            "retention_days": self.retention_days,
        }
        base_config.update(self.storage_config)
        return base_config

    def should_sample_trace(self) -> bool:
        """Determine if current trace should be sampled.

        Uses the configured sampling rate to decide whether to capture
        a trace. This helps reduce overhead in high-throughput environments.

        Returns:
            bool: True if trace should be captured, False otherwise

        Examples:
            Conditional tracing::

                if config.should_sample_trace():
                    # Capture detailed trace
                    trace_function()
        """
        import random

        return random.random() < self.trace_sampling_rate

    def get_effective_log_level(self) -> str:
        """Get the effective log level as string.

        Returns:
            str: Log level string compatible with logging libraries

        Examples:
            Configure logger::

                import logging
                logging.basicConfig(level=config.get_effective_log_level())
        """
        return self.log_level.value

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dict[str, Any]: Configuration as dictionary

        Examples:
            Serialize config::

                config_dict = config.to_dict()
                json.dump(config_dict, file)
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, (list, dict)):
                result[key] = value.copy()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        """Return string representation of configuration.

        Returns:
            str: String representation showing key settings
        """
        return (
            f"DevConfig(environment={self.environment.value}, "
            f"enabled={self.enabled}, "
            f"storage={self.storage_backend.value})"
        )


def _env_bool(key: str, default: bool) -> bool:
    """Parse boolean from environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        bool: Parsed boolean value
    """
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def _env_int(key: str, default: int) -> int:
    """Parse integer from environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        int: Parsed integer value
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    """Parse float from environment variable.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        float: Parsed float value
    """
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


# Global configuration instance
config = DevConfig.from_env()
