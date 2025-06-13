# src/haive/core/common/mixins/rich_logger_mixin.py

import logging
from typing import Optional

from pydantic import BaseModel, PrivateAttr
from rich.console import Console
from rich.logging import RichHandler


class RichLoggerMixin(BaseModel):
    """Simple mixin that provides rich logging capabilities."""

    # Debug flag to control debug output
    debug: bool = False

    # Private logger instance
    _logger: Optional[logging.Logger] = PrivateAttr(default=None)
    _logger_setup: bool = PrivateAttr(default=False)

    @property
    def logger(self) -> logging.Logger:
        """Get or create logger with rich handler."""
        if self._logger is None or not self._logger_setup:
            # Create logger with module.class name
            logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._logger = logging.getLogger(logger_name)

            # Only add handler if not already present
            if not self._logger.handlers:
                # Create rich handler with clean format
                handler = RichHandler(
                    console=Console(stderr=True),
                    show_time=True,
                    show_path=False,  # Keep it clean
                    rich_tracebacks=True,
                )
                handler.setFormatter(logging.Formatter("%(message)s"))
                self._logger.addHandler(handler)
                self._logger.setLevel(logging.INFO)

            self._logger_setup = True

        return self._logger

    def _debug(self, msg: str, *args, **kwargs) -> None:
        """Debug logging - only shows if debug=True."""
        if self.debug:
            self.logger.debug(
                f"[dim cyan]{self.__class__.__name__}:[/] {msg}", *args, **kwargs
            )

    def _info(self, msg: str, *args, **kwargs) -> None:
        """Info logging."""
        self.logger.info(msg, *args, **kwargs)

    def _warning(self, msg: str, *args, **kwargs) -> None:
        """Warning logging."""
        self.logger.warning(f"[yellow]{msg}[/]", *args, **kwargs)

    def _error(self, msg: str, *args, **kwargs) -> None:
        """Error logging."""
        self.logger.error(f"[red]{msg}[/]", *args, **kwargs)
