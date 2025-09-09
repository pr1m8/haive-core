"""State boundaries with controlled access.

This module implements BoundedState and StateView to provide controlled
access to state with explicit permissions, maintaining flexibility while
adding runtime guarantees.
"""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AccessPermissions(BaseModel):
    """Define what fields a component can access.
    
    This provides fine-grained control over state access, allowing
    different components to have different levels of access to state fields.
    
    Attributes:
        readable: Fields component can read from state
        writable: Fields component can write to state  
        append_only: Fields component can append to but not overwrite
        compute_only: Fields component can derive from but not store
        
    Examples:
        >>> # LLM engine permissions
        >>> llm_permissions = AccessPermissions(
        ...     readable={"messages", "temperature"},
        ...     writable={"response"},
        ...     append_only={"conversation_history"}
        ... )
        >>> 
        >>> # Tool node permissions  
        >>> tool_permissions = AccessPermissions(
        ...     readable={"tool_calls", "tools"},
        ...     writable={"tool_results"},
        ...     compute_only={"context"}
        ... )
    """
    
    readable: Set[str] = Field(
        default_factory=set,
        description="Fields this component can read"
    )
    writable: Set[str] = Field(
        default_factory=set,
        description="Fields this component can write"
    )
    append_only: Set[str] = Field(
        default_factory=set,
        description="Fields this component can only append to"
    )
    compute_only: Set[str] = Field(
        default_factory=set,
        description="Fields this component can use for computation but not modify"
    )
    
    def merge(self, other: AccessPermissions) -> AccessPermissions:
        """Merge with another set of permissions.
        
        Args:
            other: Other permissions to merge with
            
        Returns:
            New AccessPermissions with combined permissions
        """
        return AccessPermissions(
            readable=self.readable.union(other.readable),
            writable=self.writable.union(other.writable),
            append_only=self.append_only.union(other.append_only),
            compute_only=self.compute_only.union(other.compute_only)
        )
    
    def validate_consistency(self) -> List[str]:
        """Check for permission conflicts.
        
        Returns:
            List of consistency issues found
        """
        issues = []
        
        # Check for fields that are both writable and append_only
        both = self.writable.intersection(self.append_only)
        if both:
            issues.append(f"Fields cannot be both writable and append_only: {both}")
        
        # Check for fields that are compute_only but also writable
        compute_write = self.compute_only.intersection(self.writable)
        if compute_write:
            issues.append(f"Fields cannot be compute_only and writable: {compute_write}")
        
        return issues


class StateView:
    """Filtered view of state with access control.
    
    This provides a component with controlled access to state,
    enforcing permissions at runtime while maintaining flexibility.
    Each component gets its own view with specific permissions.
    
    Attributes:
        _state: Reference to actual state (not copied)
        _permissions: Access permissions for this view
        _access_log: Log of all access attempts
        _component_name: Name of component using this view
    """
    
    def __init__(
        self, 
        state: Dict[str, Any], 
        permissions: AccessPermissions,
        component_name: str = "unknown"
    ):
        """Initialize state view with permissions.
        
        Args:
            state: Reference to actual state (not copied)
            permissions: Access permissions for this view
            component_name: Name of component using this view
        """
        self._state = state
        self._permissions = permissions
        self._component_name = component_name
        self._access_log: List[Dict[str, Any]] = []
        self._mutations: List[Dict[str, Any]] = []
    
    def get(self, field: str, default: Any = None) -> Any:
        """Get field value with permission check.
        
        Args:
            field: Field name to retrieve
            default: Default value if field not found or not accessible
            
        Returns:
            Field value or default (deep copied for safety)
            
        Raises:
            PermissionError: If field not readable
            
        Examples:
            >>> view.get("messages")  # Returns deep copy
            >>> view.get("private_field")  # Raises PermissionError
            >>> view.get("optional", [])  # Returns default if not found
        """
        if field not in self._permissions.readable:
            self._log_access_violation("read", field)
            raise PermissionError(
                f"Component '{self._component_name}' cannot read field '{field}'"
            )
        
        self._log_access("read", field)
        value = self._state.get(field, default)
        
        # Deep copy to prevent indirect mutation
        return copy.deepcopy(value)
    
    def set(self, field: str, value: Any) -> None:
        """Set field value with permission check.
        
        Args:
            field: Field name to set
            value: Value to set
            
        Raises:
            PermissionError: If field not writable
            
        Examples:
            >>> view.set("response", "Hello")  # If writable
            >>> view.set("readonly", "value")  # Raises PermissionError
        """
        if field not in self._permissions.writable:
            self._log_access_violation("write", field)
            raise PermissionError(
                f"Component '{self._component_name}' cannot write field '{field}'"
            )
        
        # Track mutation for rollback capability
        old_value = self._state.get(field, None)
        self._mutations.append({
            "field": field,
            "old_value": copy.deepcopy(old_value),
            "new_value": copy.deepcopy(value),
            "timestamp": datetime.now().isoformat()
        })
        
        self._log_access("write", field)
        self._state[field] = value
    
    def append(self, field: str, item: Any) -> None:
        """Append to list field with permission check.
        
        Args:
            field: Field name containing list
            item: Item to append
            
        Raises:
            PermissionError: If field not appendable
            TypeError: If field is not a list
            
        Examples:
            >>> view.append("history", {"event": "processed"})
            >>> view.append("immutable_list", "item")  # Raises PermissionError
        """
        if field not in self._permissions.append_only:
            self._log_access_violation("append", field)
            raise PermissionError(
                f"Component '{self._component_name}' cannot append to field '{field}'"
            )
        
        if field not in self._state:
            self._state[field] = []
        
        if not isinstance(self._state[field], list):
            raise TypeError(f"Field '{field}' is not a list, cannot append")
        
        self._log_access("append", field)
        self._state[field].append(item)
        
        # Track append as mutation
        self._mutations.append({
            "field": field,
            "operation": "append",
            "item": copy.deepcopy(item),
            "timestamp": datetime.now().isoformat()
        })
    
    def compute_from(self, fields: List[str]) -> Dict[str, Any]:
        """Get values for computation without storage permission.
        
        Args:
            fields: Fields to retrieve for computation
            
        Returns:
            Dictionary of field values (deep copied)
            
        Raises:
            PermissionError: If any field not compute_only
            
        Examples:
            >>> # Get fields for derived computation
            >>> data = view.compute_from(["embeddings", "weights"])
            >>> result = complex_computation(data)
            >>> # Cannot store result back to those fields
        """
        result = {}
        for field in fields:
            if field not in self._permissions.compute_only:
                self._log_access_violation("compute", field)
                raise PermissionError(
                    f"Component '{self._component_name}' cannot compute from field '{field}'"
                )
            
            self._log_access("compute", field)
            result[field] = copy.deepcopy(self._state.get(field))
        
        return result
    
    def has_permission(self, operation: str, field: str) -> bool:
        """Check if operation is permitted on field.
        
        Args:
            operation: Operation type (read, write, append, compute)
            field: Field name
            
        Returns:
            True if operation is permitted
        """
        if operation == "read":
            return field in self._permissions.readable
        elif operation == "write":
            return field in self._permissions.writable
        elif operation == "append":
            return field in self._permissions.append_only
        elif operation == "compute":
            return field in self._permissions.compute_only
        else:
            return False
    
    def get_accessible_fields(self) -> Dict[str, Set[str]]:
        """Get all accessible fields by operation type.
        
        Returns:
            Dictionary mapping operation to field sets
        """
        return {
            "readable": self._permissions.readable.copy(),
            "writable": self._permissions.writable.copy(),
            "append_only": self._permissions.append_only.copy(),
            "compute_only": self._permissions.compute_only.copy()
        }
    
    def get_mutations(self) -> List[Dict[str, Any]]:
        """Get list of mutations made through this view.
        
        Returns:
            List of mutation records
        """
        return copy.deepcopy(self._mutations)
    
    def _log_access(self, operation: str, field: str) -> None:
        """Log successful access.
        
        Args:
            operation: Type of operation
            field: Field accessed
        """
        self._access_log.append({
            "timestamp": datetime.now().isoformat(),
            "component": self._component_name,
            "operation": operation,
            "field": field,
            "status": "success"
        })
    
    def _log_access_violation(self, operation: str, field: str) -> None:
        """Log access violation.
        
        Args:
            operation: Type of operation attempted
            field: Field attempted to access
        """
        self._access_log.append({
            "timestamp": datetime.now().isoformat(),
            "component": self._component_name,
            "operation": operation,
            "field": field,
            "status": "denied"
        })
        logger.warning(
            f"Access denied: Component '{self._component_name}' "
            f"attempted {operation} on field '{field}'"
        )
    
    def get_access_log(self) -> List[Dict[str, Any]]:
        """Get access log for this view.
        
        Returns:
            List of access log entries
        """
        return copy.deepcopy(self._access_log)


class BoundedState:
    """State container with access boundaries.
    
    This maintains the actual state and provides controlled views
    to different components based on their access permissions.
    Supports versioning, checkpointing, and rollback.
    
    Attributes:
        _data: The actual state data
        _access_rules: Permission rules for each component
        _version: Current state version
        _history: Checkpoint history for rollback
        _global_access_log: Consolidated access log
    """
    
    def __init__(self, initial_data: Optional[Dict[str, Any]] = None):
        """Initialize bounded state.
        
        Args:
            initial_data: Initial state data
        """
        self._data = initial_data or {}
        self._access_rules: Dict[str, AccessPermissions] = {}
        self._version = 0
        self._history: List[Dict[str, Any]] = []
        self._global_access_log: List[Dict[str, Any]] = []
        self._active_views: Dict[str, StateView] = {}
        
        # Create initial checkpoint
        self.checkpoint("Initial state")
    
    def register_component(
        self, 
        name: str, 
        permissions: AccessPermissions
    ) -> None:
        """Register component with access permissions.
        
        Args:
            name: Component identifier
            permissions: Access permissions for component
            
        Raises:
            ValueError: If permissions have conflicts
            
        Examples:
            >>> state = BoundedState()
            >>> state.register_component("llm", llm_permissions)
            >>> state.register_component("tools", tool_permissions)
        """
        # Validate permissions consistency
        issues = permissions.validate_consistency()
        if issues:
            raise ValueError(f"Permission conflicts for '{name}': {issues}")
        
        self._access_rules[name] = permissions
        logger.info(
            f"Registered component '{name}' with permissions: "
            f"read={len(permissions.readable)}, write={len(permissions.writable)}, "
            f"append={len(permissions.append_only)}, compute={len(permissions.compute_only)}"
        )
    
    def get_view_for(self, component_name: str) -> StateView:
        """Get filtered state view for component.
        
        Args:
            component_name: Name of component requesting view
            
        Returns:
            StateView with appropriate permissions
            
        Raises:
            ValueError: If component not registered
            
        Examples:
            >>> llm_view = state.get_view_for("llm")
            >>> messages = llm_view.get("messages")
            >>> llm_view.set("response", "Generated response")
        """
        if component_name not in self._access_rules:
            raise ValueError(
                f"Component '{component_name}' not registered. "
                f"Available: {list(self._access_rules.keys())}"
            )
        
        permissions = self._access_rules[component_name]
        view = StateView(self._data, permissions, component_name)
        
        # Track active views for monitoring
        self._active_views[component_name] = view
        
        return view
    
    def update_permissions(
        self, 
        component_name: str, 
        permissions: AccessPermissions
    ) -> None:
        """Update permissions for existing component.
        
        Args:
            component_name: Component to update
            permissions: New permissions
            
        Raises:
            ValueError: If component not registered
        """
        if component_name not in self._access_rules:
            raise ValueError(f"Component '{component_name}' not registered")
        
        old_permissions = self._access_rules[component_name]
        self._access_rules[component_name] = permissions
        
        logger.info(
            f"Updated permissions for '{component_name}': "
            f"readable {len(old_permissions.readable)}→{len(permissions.readable)}, "
            f"writable {len(old_permissions.writable)}→{len(permissions.writable)}"
        )
    
    def snapshot(self) -> Dict[str, Any]:
        """Get immutable snapshot of current state.
        
        Returns:
            Deep copy of current state
            
        Examples:
            >>> snapshot = state.snapshot()
            >>> # Modifications to snapshot don't affect state
            >>> snapshot["temp"] = "value"
            >>> assert "temp" not in state._data
        """
        return copy.deepcopy(self._data)
    
    def checkpoint(self, description: str = "") -> int:
        """Create checkpoint in history.
        
        Args:
            description: Optional checkpoint description
            
        Returns:
            Version number of checkpoint
            
        Examples:
            >>> version = state.checkpoint("Before processing")
            >>> # Make changes...
            >>> state.rollback(version)  # Restore to checkpoint
        """
        self._version += 1
        checkpoint = {
            "version": self._version,
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "state": copy.deepcopy(self._data),
            "access_rules": copy.deepcopy(self._access_rules)
        }
        self._history.append(checkpoint)
        
        logger.info(f"Created checkpoint v{self._version}: {description}")
        return self._version
    
    def rollback(self, version: int) -> None:
        """Rollback to previous version.
        
        Args:
            version: Version number to rollback to
            
        Raises:
            ValueError: If version not found
            
        Examples:
            >>> state.rollback(3)  # Rollback to version 3
            >>> state.rollback(state._version - 1)  # Rollback one version
        """
        for checkpoint in self._history:
            if checkpoint["version"] == version:
                self._data = copy.deepcopy(checkpoint["state"])
                self._access_rules = copy.deepcopy(checkpoint["access_rules"])
                self._version = version
                
                logger.info(
                    f"Rolled back to version {version}: {checkpoint['description']}"
                )
                return
        
        available = [c["version"] for c in self._history]
        raise ValueError(
            f"Version {version} not found. Available: {available}"
        )
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get checkpoint history.
        
        Returns:
            List of checkpoint summaries (without full state)
        """
        return [
            {
                "version": c["version"],
                "timestamp": c["timestamp"],
                "description": c["description"],
                "state_size": len(str(c["state"]))
            }
            for c in self._history
        ]
    
    def merge_access_logs(self) -> List[Dict[str, Any]]:
        """Merge access logs from all active views.
        
        Returns:
            Consolidated access log sorted by timestamp
        """
        all_logs = []
        
        for component_name, view in self._active_views.items():
            logs = view.get_access_log()
            all_logs.extend(logs)
        
        # Sort by timestamp
        all_logs.sort(key=lambda x: x["timestamp"])
        
        return all_logs
    
    def get_access_summary(self) -> Dict[str, Any]:
        """Get summary of access patterns.
        
        Returns:
            Summary statistics of field access
        """
        logs = self.merge_access_logs()
        
        # Count by operation
        operations = {}
        for log in logs:
            op = log["operation"]
            operations[op] = operations.get(op, 0) + 1
        
        # Count by field
        fields = {}
        for log in logs:
            field = log["field"]
            fields[field] = fields.get(field, 0) + 1
        
        # Count violations
        violations = sum(1 for log in logs if log["status"] == "denied")
        
        return {
            "total_accesses": len(logs),
            "operations": operations,
            "most_accessed_fields": sorted(
                fields.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10],
            "violations": violations,
            "violation_rate": violations / len(logs) if logs else 0
        }
    
    def validate_state_consistency(self) -> List[str]:
        """Check state consistency across components.
        
        Returns:
            List of consistency issues found
        """
        issues = []
        
        # Check for orphaned fields (no component can access)
        all_accessible = set()
        for permissions in self._access_rules.values():
            all_accessible.update(permissions.readable)
            all_accessible.update(permissions.writable)
            all_accessible.update(permissions.append_only)
            all_accessible.update(permissions.compute_only)
        
        orphaned = set(self._data.keys()) - all_accessible
        if orphaned:
            issues.append(f"Orphaned fields with no access rules: {orphaned}")
        
        # Check for write conflicts (multiple writers)
        write_counts = {}
        for component, permissions in self._access_rules.items():
            for field in permissions.writable:
                if field not in write_counts:
                    write_counts[field] = []
                write_counts[field].append(component)
        
        conflicts = {
            field: writers 
            for field, writers in write_counts.items() 
            if len(writers) > 1
        }
        if conflicts:
            issues.append(f"Multiple writers for fields: {conflicts}")
        
        return issues
    
    def __repr__(self) -> str:
        """String representation of bounded state."""
        return (
            f"BoundedState(version={self._version}, "
            f"fields={len(self._data)}, "
            f"components={len(self._access_rules)}, "
            f"checkpoints={len(self._history)})"
        )