#!/usr/bin/env python3
"""Test field naming utilities."""

import sys

from pydantic import BaseModel

from haive.core.schema.field_utils import field_name, get_field_info_from_model

sys.path.insert(0, "/home/will/Projects/haive/backend/haive")


# Test models
class QueryRefinementResponse(BaseModel):
    """Test model for field naming."""


@field_name("custom_query_result")
class CustomQueryResult(BaseModel):
    """Test model with custom field name."""


def test_field_naming():
    """Test field naming utilities."""
    # Test camel_to_snake_case

    # Test create_field_name_from_model

    # Test get_field_info_from_model

    return get_field_info_from_model(QueryRefinementResponse)


if __name__ == "__main__":
    field_info = test_field_naming()
