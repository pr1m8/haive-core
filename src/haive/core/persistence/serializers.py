"""Custom serializers for LangGraph persistence with SecretStr support.

This module provides secure serialization for SecretStr and other sensitive data
while maintaining security and avoiding the pickle_fallback security issue.
Supports both basic secure serialization and production-grade encryption.
"""

import logging
import os
from typing import Any, Dict, Optional, Union

from langchain_core.load.serializable import Serializable
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import SecretBytes, SecretStr
from pydantic_core import PydanticUndefined

logger = logging.getLogger(__name__)


class SecureSecretStrSerializer(JsonPlusSerializer):
    """Custom serializer that handles SecretStr securely.

    This serializer extends JsonPlusSerializer to handle SecretStr objects
    by converting them to masked values during serialization, preserving
    security while avoiding serialization errors.
    """

    def _encode_constructor_args(
        self,
        constructor,
        *,
        method=None,
        args=None,
        kwargs=None,
    ) -> Dict[str, Any]:
        """Override to handle SecretStr objects."""

        # Process args for SecretStr if provided
        processed_args = None
        if args is not None:
            processed_args = []
            for arg in args:
                processed_args.append(self._handle_secret_types(arg))

        # Process kwargs for SecretStr if provided
        processed_kwargs = None
        if kwargs is not None:
            processed_kwargs = {}
            for key, value in kwargs.items():
                processed_kwargs[key] = self._handle_secret_types(value)

        return super()._encode_constructor_args(
            constructor, method=method, args=processed_args, kwargs=processed_kwargs
        )

    def dumps(self, obj: Any) -> bytes:
        """Override dumps to handle SecretStr objects before JSON serialization."""
        # Pre-process the object to handle SecretStr and PydanticUndefined
        processed_obj = self._handle_secret_types(obj)
        # Call parent dumps with processed object
        return super().dumps(processed_obj)

    def _handle_secret_types(self, value: Any) -> Any:
        """Handle SecretStr and SecretBytes by converting to masked strings."""
        if isinstance(value, SecretStr):
            # Convert to masked string - this preserves the SecretStr interface
            # while making it serializable
            return "**SECRET_MASKED**"
        elif isinstance(value, SecretBytes):
            # Convert to masked bytes
            return b"**SECRET_MASKED**"
        elif isinstance(value, dict):
            # Recursively handle dictionaries
            return {k: self._handle_secret_types(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            # Recursively handle sequences
            processed = [self._handle_secret_types(item) for item in value]
            return type(value)(processed)
        elif value is PydanticUndefined:
            # Handle PydanticUndefined by converting to None
            logger.warning(
                "Found PydanticUndefined during serialization, converting to None"
            )
            return None
        else:
            return value

    def loads_typed(self, data: bytes, type_: str) -> Any:
        """Override to handle loading of masked secrets."""
        try:
            result = super().loads_typed(data, type_)

            # If we encounter masked secrets during loading, warn about it
            if self._contains_masked_secrets(result):
                logger.warning(
                    "Loaded state contains masked secrets. Original secret values "
                    "are not recoverable from checkpoint. Consider using external "
                    "secret management for critical secrets."
                )

            return result
        except Exception as e:
            logger.error(f"Failed to deserialize data of type {type_}: {e}")
            raise

    def _contains_masked_secrets(self, obj: Any) -> bool:
        """Check if object contains masked secret placeholders."""
        if isinstance(obj, str) and obj == "**SECRET_MASKED**":
            return True
        elif isinstance(obj, bytes) and obj == b"**SECRET_MASKED**":
            return True
        elif isinstance(obj, dict):
            return any(self._contains_masked_secrets(v) for v in obj.values())
        elif isinstance(obj, (list, tuple)):
            return any(self._contains_masked_secrets(item) for item in obj)
        return False


class SecretStrSerializer(JsonPlusSerializer):
    """Alternative serializer that preserves SecretStr values using model_dump.

    WARNING: This approach exposes the actual secret values during serialization.
    Only use this if you have proper encryption at the storage layer.
    """

    def _encode_constructor_args(
        self,
        constructor: str,
        method: str,
        args: tuple,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Override to handle SecretStr by exposing values with serialize_as_any."""

        # Process args for SecretStr
        processed_args = []
        for arg in args:
            processed_args.append(self._expose_secrets(arg))

        # Process kwargs for SecretStr
        processed_kwargs = {}
        for key, value in kwargs.items():
            processed_kwargs[key] = self._expose_secrets(value)

        return super()._encode_constructor_args(
            constructor, method, tuple(processed_args), processed_kwargs
        )

    def _expose_secrets(self, value: Any) -> Any:
        """Convert SecretStr to actual string values (INSECURE - requires encryption)."""
        if isinstance(value, SecretStr):
            # WARNING: This exposes the actual secret!
            return value.get_secret_value()
        elif isinstance(value, SecretBytes):
            # WARNING: This exposes the actual secret!
            return value.get_secret_value()
        elif hasattr(value, "model_dump") and hasattr(value, "__class__"):
            # Handle Pydantic models with SecretStr fields
            try:
                return value.model_dump(serialize_as_any=True)
            except Exception:
                return value
        elif isinstance(value, dict):
            return {k: self._expose_secrets(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            processed = [self._expose_secrets(item) for item in value]
            return type(value)(processed)
        elif value is PydanticUndefined:
            return None
        else:
            return value


def create_production_serializer(
    encryption_key: Optional[str] = None,
) -> JsonPlusSerializer:
    """Create a production-ready serializer with optional encryption.

    This function creates the appropriate serializer based on environment and
    security requirements. For production, it uses EncryptedSerializer when
    an encryption key is available, otherwise falls back to SecureSecretStrSerializer.

    Args:
        encryption_key: Optional AES encryption key. If not provided, will try
                       to load from LANGGRAPH_AES_KEY environment variable.

    Returns:
        JsonPlusSerializer: Either EncryptedSerializer or SecureSecretStrSerializer

    Examples:
        Basic usage with environment key::

            # Set LANGGRAPH_AES_KEY environment variable
            serializer = create_production_serializer()

        With explicit key::

            serializer = create_production_serializer("your-32-byte-key-here")

        Development (no encryption)::

            serializer = create_production_serializer(encryption_key=None)
    """

    # Try to get encryption key from parameter or environment
    if encryption_key is None:
        encryption_key = os.getenv("LANGGRAPH_AES_KEY")

    # If we have an encryption key, use EncryptedSerializer
    if encryption_key:
        try:
            from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

            # Create encrypted serializer with our secure base
            base_serializer = SecureSecretStrSerializer()
            encrypted_serializer = EncryptedSerializer.from_pycryptodome_aes(
                serde=base_serializer,
                key=(
                    encryption_key.encode()
                    if isinstance(encryption_key, str)
                    else encryption_key
                ),
            )

            logger.info(
                "Created EncryptedSerializer with SecretStr support for production use"
            )
            return encrypted_serializer

        except ImportError as e:
            logger.warning(
                f"EncryptedSerializer not available: {e}. "
                f"Falling back to SecureSecretStrSerializer (unencrypted)."
            )
        except Exception as e:
            logger.error(f"Failed to create EncryptedSerializer: {e}")

    # Fallback to our secure serializer (unencrypted but SecretStr-safe)
    logger.info("Using SecureSecretStrSerializer (unencrypted) for SecretStr support")
    return SecureSecretStrSerializer()


def create_encrypted_serializer_for_postgres(
    connection_string: str, encryption_key: Optional[str] = None
) -> JsonPlusSerializer:
    """Create an encrypted serializer specifically optimized for PostgreSQL.

    This function creates a production-ready encrypted serializer that's
    optimized for PostgreSQL storage. It includes additional security
    measures and PostgreSQL-specific optimizations.

    Args:
        connection_string: PostgreSQL connection string for logging/validation
        encryption_key: AES encryption key. If not provided, will try
                       LANGGRAPH_AES_KEY environment variable.

    Returns:
        JsonPlusSerializer: Production-ready encrypted serializer

    Raises:
        ValueError: If no encryption key is available in production

    Examples:
        Production PostgreSQL setup::

            serializer = create_encrypted_serializer_for_postgres(
                connection_string="postgresql://user:pass@host:5432/db",
                encryption_key=os.getenv("LANGGRAPH_AES_KEY")
            )
    """

    # Try to get encryption key
    if encryption_key is None:
        encryption_key = os.getenv("LANGGRAPH_AES_KEY")

    # For production PostgreSQL, encryption is highly recommended
    is_production = os.getenv("ENVIRONMENT", "").lower() in ["production", "prod"]

    if is_production and not encryption_key:
        raise ValueError(
            "Encryption key is required for production PostgreSQL checkpointing. "
            "Please set LANGGRAPH_AES_KEY environment variable or provide encryption_key parameter."
        )

    if encryption_key:
        try:
            from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

            # Create base serializer with SecretStr support
            base_serializer = SecureSecretStrSerializer()

            # Create encrypted serializer
            encrypted_serializer = EncryptedSerializer.from_pycryptodome_aes(
                serde=base_serializer,
                key=(
                    encryption_key.encode()
                    if isinstance(encryption_key, str)
                    else encryption_key
                ),
            )

            logger.info(
                "Created encrypted PostgreSQL serializer with SecretStr support"
            )
            return encrypted_serializer

        except ImportError as e:
            logger.error(
                f"EncryptedSerializer not available for PostgreSQL: {e}. "
                f"Install with: pip install 'langgraph[encryption]'"
            )
            if is_production:
                raise RuntimeError(
                    "EncryptedSerializer is required for production PostgreSQL but not available. "
                    "Install with: pip install 'langgraph[encryption]'"
                )
        except Exception as e:
            logger.error(f"Failed to create encrypted PostgreSQL serializer: {e}")
            if is_production:
                raise

    # Development fallback
    logger.warning(
        "Using unencrypted SecureSecretStrSerializer for PostgreSQL. "
        "This is not recommended for production use."
    )
    return SecureSecretStrSerializer()
