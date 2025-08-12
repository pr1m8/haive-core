"""
Development context manager for unified operations.

Provides a unified context manager for debugging, logging, tracing,
and profiling operations with correlation ID management.
"""

import time
import uuid
from typing import Any

from haive.core.utils.debugkit.config import config


class DevContext:
    """Context manager for unified development operations.

    Provides a unified interface for logging, tracing, profiling,
    and debugging within a named context. Automatically manages
    correlation IDs, timing, and resource cleanup.

    Attributes:
        name: Name of the context operation
        correlation_id: Unique identifier for this context
        start_time: When the context was started
        data: Additional data collected during context execution

    Examples:
        Basic context usage::

            with debugkit.context("user_registration") as ctx:
                ctx.debug("Starting registration", user_id=user.id)

                # Mark checkpoints
                ctx.checkpoint("validation_complete")

                # Record metrics
                ctx.record("users_processed", 1)

                ctx.success("Registration complete")

        Nested contexts::

            with debugkit.context("api_request") as outer:
                with debugkit.context("database_query") as inner:
                    inner.debug("Executing query", table="users")
                    # Inner context inherits correlation_id

                outer.info("Request processed")
    """

    def __init__(self, name: str, correlation_id: str | None = None, **metadata: Any):
        """Initialize development context.

        Args:
            name: Name of the context operation
            correlation_id: Optional correlation ID (generated if None)
            **metadata: Additional metadata for the context
        """
        self.name = name
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.metadata = metadata
        self.start_time: float | None = None
        self.data: dict[str, Any] = {}
        self._checkpoints: list[dict[str, Any]] = []

    def __enter__(self) -> "DevContext":
        """Enter the context."""
        self.start_time = time.time()

        # Import here to avoid circular imports
        from haive.core.utils.debugkit.logging import log
        from haive.core.utils.debugkit.tracing import trace

        # Set correlation ID in components
        if hasattr(log, "set_correlation_id"):
            log.set_correlation_id(self.correlation_id)
        if hasattr(trace, "set_correlation_id"):
            trace.set_correlation_id(self.correlation_id)

        # Log context entry
        if config.log_enabled:
            log.info(
                f"Entering context: {self.name}",
                correlation_id=self.correlation_id,
                **self.metadata,
            )

        # Start tracing if enabled
        if config.trace_enabled and hasattr(trace, "push_context"):
            trace.push_context(self.name, self.correlation_id)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        duration = time.time() - self.start_time if self.start_time else 0.0

        # Import here to avoid circular imports
        from haive.core.utils.debugkit.logging import log
        from haive.core.utils.debugkit.tracing import trace

        # Log context exit
        if config.log_enabled:
            if exc_type is None:
                log.info(
                    f"Exiting context: {self.name}",
                    correlation_id=self.correlation_id,
                    duration=duration,
                    checkpoints=len(self._checkpoints),
                    **self.data,
                )
            else:
                log.error(
                    f"Context failed: {self.name}",
                    correlation_id=self.correlation_id,
                    duration=duration,
                    error=str(exc_val),
                    **self.data,
                )

        # Pop trace context
        if config.trace_enabled and hasattr(trace, "pop_context"):
            trace.pop_context()

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message within context.

        Args:
            message: Debug message
            **kwargs: Additional context data
        """
        if config.debug_enabled:
            from haive.core.utils.debugkit.debug import debug

            debug.ice(
                f"[{self.name}] {message}", correlation_id=self.correlation_id, **kwargs
            )

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message within context.

        Args:
            message: Info message
            **kwargs: Additional context data
        """
        if config.log_enabled:
            from haive.core.utils.debugkit.logging import log

            log.info(
                f"[{self.name}] {message}", correlation_id=self.correlation_id, **kwargs
            )

    def success(self, message: str, **kwargs: Any) -> None:
        """Log success message within context.

        Args:
            message: Success message
            **kwargs: Additional context data
        """
        if config.log_enabled:
            from haive.core.utils.debugkit.logging import log

            log.success(
                f"[{self.name}] {message}", correlation_id=self.correlation_id, **kwargs
            )

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message within context.

        Args:
            message: Error message
            **kwargs: Additional context data
        """
        if config.log_enabled:
            from haive.core.utils.debugkit.logging import log

            log.error(
                f"[{self.name}] {message}", correlation_id=self.correlation_id, **kwargs
            )

    def checkpoint(self, checkpoint_name: str, **data: Any) -> None:
        """Mark a checkpoint within the context.

        Args:
            checkpoint_name: Name of the checkpoint
            **data: Additional checkpoint data
        """
        elapsed = time.time() - self.start_time if self.start_time else 0.0

        checkpoint_data = {
            "name": checkpoint_name,
            "elapsed_time": elapsed,
            "timestamp": time.time(),
            **data,
        }

        self._checkpoints.append(checkpoint_data)

        if config.trace_enabled:
            from haive.core.utils.debugkit.tracing import trace

            if hasattr(trace, "mark"):
                trace.mark(f"{self.name}.{checkpoint_name}", elapsed)

        if config.verbose:
            self.debug(f"Checkpoint: {checkpoint_name}", elapsed=elapsed, **data)

    def record(self, key: str, value: Any) -> None:
        """Record data within the context.

        Args:
            key: Data key
            value: Data value
        """
        self.data[key] = value

    def get_elapsed_time(self) -> float:
        """Get elapsed time since context start.

        Returns:
            float: Elapsed time in seconds
        """
        return time.time() - self.start_time if self.start_time else 0.0
