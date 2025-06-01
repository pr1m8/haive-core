"""
Logging configuration utility for clean, configurable logging.

This module provides utilities for managing logging in game agents and other components,
with support for different verbosity levels and output formats.
"""

import logging
import os
import sys
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

# Try to import rich for enhanced console output
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


class LogLevel(str, Enum):
    """Logging levels"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SILENT = "SILENT"  # No logging except critical errors


class LogFormat(str, Enum):
    """Log output formats"""

    SIMPLE = "simple"  # Basic text output
    DETAILED = "detailed"  # Include timestamps and module info
    JSON = "json"  # JSON formatted logs
    RICH = "rich"  # Rich console output (if available)


class GameLogger:
    """
    Enhanced logger for game agents with clean output and debugging control.

    Features:
    - Configurable verbosity levels
    - Rich console output support
    - Game-specific formatting
    - Performance tracking
    """

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        format: LogFormat = LogFormat.RICH if RICH_AVAILABLE else LogFormat.SIMPLE,
        enable_file_logging: bool = False,
        log_file: Optional[str] = None,
    ):
        """
        Initialize the game logger.

        Args:
            name: Logger name (usually module name)
            level: Logging level
            format: Output format
            enable_file_logging: Whether to log to file
            log_file: Path to log file
        """
        self.name = name
        self.level = level
        self.format = format
        self.enable_file_logging = enable_file_logging
        self.log_file = (
            log_file or f"game_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        # Performance tracking
        self._operation_starts: Dict[str, float] = {}

        # Set up the logger
        self._setup_logger()

    def _setup_logger(self):
        """Set up the logging configuration"""
        self.logger = logging.getLogger(self.name)

        # Clear existing handlers
        self.logger.handlers = []

        # Set level
        if self.level == LogLevel.SILENT:
            self.logger.setLevel(logging.CRITICAL + 1)
        else:
            self.logger.setLevel(getattr(logging, self.level.value))

        # Console handler
        if self.format == LogFormat.RICH and RICH_AVAILABLE:
            console_handler = RichHandler(
                console=console, show_time=False, show_path=False, markup=True
            )
        else:
            console_handler = logging.StreamHandler(sys.stdout)

        # Set formatter based on format type
        if self.format == LogFormat.SIMPLE:
            formatter = logging.Formatter("%(message)s")
        elif self.format == LogFormat.DETAILED:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        elif self.format == LogFormat.JSON:
            # JSON formatter would go here
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "name": "%(name)s", '
                '"level": "%(levelname)s", "message": "%(message)s"}',
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        else:
            formatter = logging.Formatter("%(message)s")

        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler if enabled
        if self.enable_file_logging:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self.logger.addHandler(file_handler)

    def set_level(self, level: LogLevel):
        """Change the logging level"""
        self.level = level
        if level == LogLevel.SILENT:
            self.logger.setLevel(logging.CRITICAL + 1)
        else:
            self.logger.setLevel(getattr(logging, level.value))

    # Game-specific logging methods

    def turn_start(self, player_name: str, turn_number: int, **kwargs):
        """Log the start of a player's turn"""
        if RICH_AVAILABLE and self.format == LogFormat.RICH:
            console.print(
                f"\n[bold cyan]🎲 Turn {turn_number}:[/bold cyan] [yellow]{player_name}[/yellow]"
            )
            if kwargs:
                for key, value in kwargs.items():
                    console.print(f"  {key}: {value}")
        else:
            self.logger.info(f"Turn {turn_number}: {player_name}'s turn")
            if kwargs:
                for key, value in kwargs.items():
                    self.logger.info(f"  {key}: {value}")

    def dice_roll(self, player_name: str, die1: int, die2: int, total: int):
        """Log a dice roll"""
        if RICH_AVAILABLE and self.format == LogFormat.RICH:
            console.print(
                f"  🎲 [blue]{player_name}[/blue] rolled: {die1} + {die2} = [bold]{total}[/bold]"
            )
        else:
            self.logger.info(f"{player_name} rolled: {die1} + {die2} = {total}")

    def player_move(
        self, player_name: str, from_pos: int, to_pos: int, passed_go: bool = False
    ):
        """Log player movement"""
        if RICH_AVAILABLE and self.format == LogFormat.RICH:
            move_text = (
                f"  🚶 [blue]{player_name}[/blue] moves from {from_pos} to {to_pos}"
            )
            if passed_go:
                move_text += " [bold green](Passed GO! +$200)[/bold green]"
            console.print(move_text)
        else:
            msg = f"{player_name} moves from {from_pos} to {to_pos}"
            if passed_go:
                msg += " (Passed GO! +$200)"
            self.logger.info(msg)

    def property_action(
        self,
        action: str,
        player_name: str,
        property_name: str,
        amount: Optional[int] = None,
        **kwargs,
    ):
        """Log property-related actions"""
        icons = {
            "landed": "🎯",
            "buy": "✅",
            "pass": "🚫",
            "rent": "💸",
            "mortgage": "🏦",
            "unmortgage": "🏠",
            "build": "🏗️",
        }

        icon = icons.get(action, "📍")

        if RICH_AVAILABLE and self.format == LogFormat.RICH:
            if action == "buy":
                console.print(
                    f"  {icon} [green]{player_name} purchased {property_name} for ${amount}[/green]"
                )
            elif action == "rent":
                recipient = kwargs.get("recipient", "unknown")
                console.print(
                    f"  {icon} [red]{player_name} paid ${amount} rent to {recipient}[/red]"
                )
            else:
                msg = f"  {icon} {player_name} {action} {property_name}"
                if amount:
                    msg += f" (${amount})"
                console.print(msg)
        else:
            msg = f"{player_name} {action} {property_name}"
            if amount:
                msg += f" (${amount})"
            self.logger.info(msg)

    def game_event(self, event_type: str, description: str, **kwargs):
        """Log a general game event"""
        if RICH_AVAILABLE and self.format == LogFormat.RICH:
            console.print(f"  📌 [magenta]{event_type}:[/magenta] {description}")
            if kwargs and self.logger.level <= logging.DEBUG:
                for key, value in kwargs.items():
                    console.print(f"    • {key}: {value}")
        else:
            self.logger.info(f"{event_type}: {description}")
            if kwargs and self.logger.level <= logging.DEBUG:
                for key, value in kwargs.items():
                    self.logger.debug(f"  {key}: {value}")

    def decision(
        self, player_name: str, decision_type: str, choice: str, reasoning: str = ""
    ):
        """Log a player decision"""
        if RICH_AVAILABLE and self.format == LogFormat.RICH:
            console.print(
                f"  🤔 [yellow]{player_name}[/yellow] decides to [bold]{choice}[/bold]"
            )
            if reasoning and self.logger.level <= logging.DEBUG:
                console.print(f"     [dim]{reasoning}[/dim]")
        else:
            self.logger.info(f"{player_name} decides to {choice}")
            if reasoning and self.logger.level <= logging.DEBUG:
                self.logger.debug(f"  Reasoning: {reasoning}")

    def game_state_summary(self, state: Dict[str, Any]):
        """Display a summary of the game state"""
        if (
            RICH_AVAILABLE
            and self.format == LogFormat.RICH
            and self.logger.level <= logging.DEBUG
        ):
            table = Table(title="Game State Summary")
            table.add_column("Player", style="cyan")
            table.add_column("Money", style="green")
            table.add_column("Properties", style="yellow")
            table.add_column("Status", style="magenta")

            players = state.get("players", [])
            for player in players:
                status = "Bankrupt" if player.get("bankrupt", False) else "Active"
                if player.get("in_jail", False):
                    status = "In Jail"

                table.add_row(
                    player.get("name", "Unknown"),
                    f"${player.get('money', 0)}",
                    str(len(player.get("properties", []))),
                    status,
                )

            console.print(table)
        elif self.logger.level <= logging.DEBUG:
            self.logger.debug("Game State Summary:")
            players = state.get("players", [])
            for player in players:
                self.logger.debug(
                    f"  {player.get('name')}: ${player.get('money')} | "
                    f"Properties: {len(player.get('properties', []))}"
                )

    def performance_start(self, operation: str):
        """Start timing an operation"""
        import time

        self._operation_starts[operation] = time.time()

    def performance_end(self, operation: str):
        """End timing an operation and log if in debug mode"""
        import time

        if operation in self._operation_starts:
            duration = time.time() - self._operation_starts[operation]
            del self._operation_starts[operation]

            if self.logger.level <= logging.DEBUG:
                if RICH_AVAILABLE and self.format == LogFormat.RICH:
                    console.print(f"  ⏱️  [dim]{operation} took {duration:.3f}s[/dim]")
                else:
                    self.logger.debug(f"{operation} took {duration:.3f}s")

    # Standard logging methods
    def debug(self, msg: str, **kwargs):
        """Log debug message"""
        self.logger.debug(msg, **kwargs)

    def info(self, msg: str, **kwargs):
        """Log info message"""
        self.logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        """Log warning message"""
        self.logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs):
        """Log error message"""
        self.logger.error(msg, **kwargs)

    def critical(self, msg: str, **kwargs):
        """Log critical message"""
        self.logger.critical(msg, **kwargs)


def get_game_logger(
    name: str, level: Optional[str] = None, format: Optional[str] = None
) -> GameLogger:
    """
    Get a configured game logger instance.

    Args:
        name: Logger name
        level: Override log level from environment
        format: Override format from environment

    Returns:
        Configured GameLogger instance
    """
    # Check environment variables for configuration
    env_level = level or os.getenv("GAME_LOG_LEVEL", "INFO")
    env_format = format or os.getenv(
        "GAME_LOG_FORMAT", "rich" if RICH_AVAILABLE else "simple"
    )
    enable_file = os.getenv("GAME_LOG_TO_FILE", "false").lower() == "true"

    # Convert to enums
    try:
        log_level = LogLevel[env_level.upper()]
    except KeyError:
        log_level = LogLevel.INFO

    try:
        log_format = (
            LogFormat[env_format.upper()]
            if env_format.upper() in LogFormat.__members__
            else LogFormat(env_format.lower())
        )
    except (KeyError, ValueError):
        log_format = LogFormat.RICH if RICH_AVAILABLE else LogFormat.SIMPLE

    return GameLogger(
        name=name, level=log_level, format=log_format, enable_file_logging=enable_file
    )
