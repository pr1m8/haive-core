# PostgreSQL Security Guide for Haive Framework

## Overview

This guide explains the comprehensive security approach for PostgreSQL persistence in the Haive framework, including SecretStr handling, encryption, and production best practices.

## Security Architecture

### 1. Multi-Layer Security Approach

```
Application Layer (SecretStr)
    ↓
Serialization Layer (Custom Serializer)
    ↓
Encryption Layer (EncryptedSerializer)
    ↓
Transport Layer (SSL/TLS)
    ↓
Database Layer (PostgreSQL encryption)
```

### 2. SecretStr Handling

The framework provides secure handling of Pydantic SecretStr fields:

- **SecureSecretStrSerializer**: Converts SecretStr to masked placeholders during serialization
- **Preserves Security**: Never exposes actual secret values in checkpoint data
- **Serialization Safety**: Prevents msgpack serialization errors

### 3. Encryption Options

#### Development (Basic Security)

```python
# Uses SecureSecretStrSerializer (unencrypted but SecretStr-safe)
config = PostgresCheckpointerConfig(
    db_host="localhost",
    db_port=5432,
    db_name="haive_dev"
)
```

#### Production (Full Encryption)

```python
# Set environment variable for encryption
os.environ["LANGGRAPH_AES_KEY"] = "your-32-byte-encryption-key"

# Uses EncryptedSerializer with SecretStr support
config = PostgresCheckpointerConfig(
    db_host="prod-db.example.com",
    db_port=5432,
    db_name="haive_prod",
    ssl_mode="require"
)
```

## Implementation Details

### Custom Serializers

#### SecureSecretStrSerializer

- Handles SecretStr by converting to masked strings
- Prevents accidental exposure of secrets
- Safe fallback for PydanticUndefined values

#### EncryptedSerializer (Production)

- Uses AES encryption for all checkpoint data
- Layered on top of SecureSecretStrSerializer
- Requires LANGGRAPH_AES_KEY environment variable

### Production Setup

#### 1. Environment Variables

```bash
# Required for encryption
export LANGGRAPH_AES_KEY="your-32-byte-aes-encryption-key"

# PostgreSQL connection
export POSTGRES_CONNECTION_STRING="postgresql://user:pass@host:5432/db?sslmode=require"

# Environment detection
export ENVIRONMENT="production"
```

#### 2. Connection Security

```python
config = PostgresCheckpointerConfig(
    connection_string=os.getenv("POSTGRES_CONNECTION_STRING"),
    ssl_mode="require",  # Force SSL/TLS
    prepare_threshold=None,  # Disable prepared statements
    auto_commit=True,
    min_pool_size=1,
    max_pool_size=10
)
```

#### 3. Database-Level Security

```sql
-- Enable PostgreSQL extensions for additional security
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Create dedicated user with limited permissions
CREATE USER haive_app WITH PASSWORD 'secure_password';
GRANT CONNECT ON DATABASE haive_prod TO haive_app;
GRANT USAGE ON SCHEMA public TO haive_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON checkpoints TO haive_app;
```

## Security Best Practices

### 1. Encryption Key Management

- Use 32-byte random keys for AES encryption
- Store keys in secure key management systems (AWS KMS, Azure Key Vault, etc.)
- Rotate keys regularly
- Never commit keys to version control

### 2. Database Security

- Use SSL/TLS for all connections (sslmode=require)
- Implement network-level security (VPC, firewalls)
- Use dedicated database users with minimal permissions
- Enable PostgreSQL audit logging

### 3. Application Security

- Validate all SecretStr fields at application boundaries
- Use environment variables for sensitive configuration
- Implement proper error handling to prevent information leakage
- Regular security audits of checkpoint data

### 4. Monitoring and Compliance

- Log security events (encryption/decryption failures)
- Monitor unusual access patterns
- Implement data retention policies
- Ensure compliance with regulations (GDPR, HIPAA, etc.)

## Example Configurations

### Development Environment

```python
from haive.core.persistence.postgres_config import PostgresCheckpointerConfig

# Basic security with SecretStr masking
config = PostgresCheckpointerConfig(
    db_host="localhost",
    db_port=5432,
    db_name="haive_dev",
    db_user="haive_dev",
    db_pass=SecretStr("dev_password"),
    ssl_mode="disable"  # OK for development
)

checkpointer = config.create_checkpointer()
```

### Production Environment

```python
from haive.core.persistence.postgres_config import PostgresCheckpointerConfig
import os

# Full encryption + SSL/TLS
config = PostgresCheckpointerConfig(
    connection_string=os.getenv("POSTGRES_CONNECTION_STRING"),
    ssl_mode="require",
    prepare_threshold=None,
    auto_commit=True,
    connection_kwargs={
        "application_name": "haive_prod",
        "connect_timeout": 30,
        "keepalives": 1,
        "keepalives_idle": 30
    }
)

# Requires LANGGRAPH_AES_KEY environment variable
checkpointer = config.create_checkpointer()
```

### Multi-Tenant Production

```python
# Per-tenant encryption keys
def create_tenant_checkpointer(tenant_id: str):
    encryption_key = get_tenant_encryption_key(tenant_id)  # From key management system

    config = PostgresCheckpointerConfig(
        connection_string=os.getenv("POSTGRES_CONNECTION_STRING"),
        ssl_mode="require"
    )

    # Custom serializer with tenant-specific encryption
    from haive.core.persistence.serializers import create_encrypted_serializer_for_postgres
    serializer = create_encrypted_serializer_for_postgres(
        connection_string=config.get_connection_uri(),
        encryption_key=encryption_key
    )

    return config.create_checkpointer()
```

## Security Validation

### Test Security Implementation

```python
import pytest
from pydantic import SecretStr
from haive.core.persistence.serializers import SecureSecretStrSerializer

def test_secret_str_serialization():
    """Verify SecretStr values are properly masked."""
    serializer = SecureSecretStrSerializer()

    test_data = {
        "api_key": SecretStr("sk-secret-key-123"),
        "regular_field": "normal_value"
    }

    # Serialize and check for masked secrets
    serialized = serializer._handle_secret_types(test_data)

    assert serialized["api_key"] == "**SECRET_MASKED**"
    assert serialized["regular_field"] == "normal_value"

def test_production_encryption():
    """Verify production encryption works correctly."""
    # Test with actual encryption key
    pass
```

## Troubleshooting

### Common Issues

1. **SecretStr Serialization Errors**
   - Ensure using custom serializers
   - Check for PydanticUndefined handling

2. **Encryption Key Issues**
   - Verify LANGGRAPH_AES_KEY is set correctly
   - Check key length (must be 32 bytes for AES-256)

3. **Connection Issues**
   - Verify SSL/TLS configuration
   - Check firewall and network settings
   - Validate connection string format

### Debug Commands

```python
# Test serializer
from haive.core.persistence.serializers import create_production_serializer
serializer = create_production_serializer()
print(f"Serializer type: {type(serializer)}")

# Test connection
config = PostgresCheckpointerConfig(...)
try:
    checkpointer = config.create_checkpointer()
    print("✅ Connection successful")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

## Compliance Notes

- **GDPR**: Encryption ensures personal data protection
- **HIPAA**: Meets encryption requirements for PHI
- **SOC 2**: Supports data security controls
- **ISO 27001**: Aligns with information security standards

## Migration Guide

### From Basic to Encrypted

1. Set up encryption key management
2. Set LANGGRAPH_AES_KEY environment variable
3. Restart application (automatic detection)
4. Verify encryption is active in logs

### From Development to Production

1. Enable SSL/TLS on PostgreSQL
2. Update connection strings with SSL parameters
3. Set ENVIRONMENT=production
4. Configure encryption keys
5. Update monitoring and alerting
