"""Tests for FieldMapping dataclass - using real components only."""

from haive.core.graph.node.composer import FieldMapping


class TestFieldMapping:
    """Test FieldMapping dataclass behavior."""

    def test_basic_field_mapping_creation(self):
        """Test creating basic field mapping."""
        mapping = FieldMapping(source_path="result", target_path="potato")

        assert mapping.source_path == "result"
        assert mapping.target_path == "potato"
        assert mapping.transform is None
        assert mapping.default is None
        assert mapping.required is False

    def test_field_mapping_with_transform(self):
        """Test field mapping with transform list."""
        mapping = FieldMapping(
            source_path="content", target_path="text", transform=["strip", "uppercase"]
        )

        assert mapping.source_path == "content"
        assert mapping.target_path == "text"
        assert mapping.transform == ["strip", "uppercase"]
        assert mapping.default is None
        assert mapping.required is False

    def test_field_mapping_with_default(self):
        """Test field mapping with default value."""
        mapping = FieldMapping(
            source_path="temperature", target_path="temp", default=0.7
        )

        assert mapping.source_path == "temperature"
        assert mapping.target_path == "temp"
        assert mapping.transform is None
        assert mapping.default == 0.7
        assert mapping.required is False

    def test_field_mapping_required(self):
        """Test required field mapping."""
        mapping = FieldMapping(
            source_path="messages", target_path="conversation", required=True
        )

        assert mapping.source_path == "messages"
        assert mapping.target_path == "conversation"
        assert mapping.transform is None
        assert mapping.default is None
        assert mapping.required is True

    def test_field_mapping_all_options(self):
        """Test field mapping with all options set."""
        mapping = FieldMapping(
            source_path="messages[-1].content",
            target_path="last_message",
            transform=["strip", "lowercase"],
            default="",
            required=False,
        )

        assert mapping.source_path == "messages[-1].content"
        assert mapping.target_path == "last_message"
        assert mapping.transform == ["strip", "lowercase"]
        assert mapping.default == ""
        assert mapping.required is False

    def test_field_mapping_equality(self):
        """Test field mapping equality comparison."""
        mapping1 = FieldMapping("source", "target")
        mapping2 = FieldMapping("source", "target")
        mapping3 = FieldMapping("source", "different")

        assert mapping1 == mapping2
        assert mapping1 != mapping3

    def test_field_mapping_as_dict(self):
        """Test converting field mapping to dict-like representation."""
        mapping = FieldMapping(
            source_path="input",
            target_path="output",
            transform=["parse_json"],
            default={},
            required=True,
        )

        # Dataclasses support __dict__
        mapping_dict = mapping.__dict__

        assert mapping_dict["source_path"] == "input"
        assert mapping_dict["target_path"] == "output"
        assert mapping_dict["transform"] == ["parse_json"]
        assert mapping_dict["default"] == {}
        assert mapping_dict["required"] is True
