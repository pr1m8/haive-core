# Tree/Leaf Structure Module

An enhanced tree structure implementation with advanced generic support for the Haive framework.

## Overview

The `tree_leaf` module provides a powerful generic tree structure that improves upon the existing AutoTree with:

- **Multiple TypeVars**: Separate types for content, children, and results
- **Better Type Inference**: Default TypeVars and bounded generics
- **Auto-Indexing**: Automatic path tracking and indexing
- **Computed Properties**: Dynamic properties like node counts, progress tracking
- **Heterogeneous Trees**: Support for mixed child types

## Key Components

### Base Classes

- **`TreeNode`**: Abstract base class for all nodes
- **`Leaf`**: Terminal nodes with content but no children
- **`Tree`**: Branch nodes with content and children

### Type Variables

- **`ContentT`**: What each node contains (bounded to BaseModel)
- **`ChildT`**: Type of children a tree can have (bounded to BaseModel)
- **`ResultT`**: What executing a node produces (defaults to Any)

### Default Types

- **`DefaultContent`**: Simple content with name/value
- **`DefaultResult`**: Result with success/data/error

## Usage Examples

### Basic Tree Creation

```python
from haive.core.common.structures import Tree, Leaf, DefaultContent

# Create a tree
root = Tree(content=DefaultContent(name="Project"))

# Add leaves
root.add_child(Leaf(content=DefaultContent(name="Task 1")))
root.add_child(Leaf(content=DefaultContent(name="Task 2")))

# Add subtree
subtree = Tree(content=DefaultContent(name="Phase 2"))
subtree.add_child(Leaf(content=DefaultContent(name="Subtask 2.1")))
root.add_child(subtree)
```

### Typed Trees

```python
from pydantic import BaseModel
from haive.core.common.structures import Tree, Leaf

class TaskContent(BaseModel):
    name: str
    priority: int

class TaskResult(BaseModel):
    completed: bool
    output: str

# Create typed tree
task_tree: Tree[TaskContent, Tree[TaskContent, TaskResult], TaskResult] = Tree(
    content=TaskContent(name="Main Task", priority=1)
)

# Add typed leaf
leaf: Leaf[TaskContent, TaskResult] = Leaf(
    content=TaskContent(name="Subtask", priority=2)
)
task_tree.add_child(leaf)
```

### Computed Properties

```python
# Access computed properties
print(f"Total nodes: {tree.descendant_count}")
print(f"Tree height: {tree.height}")
print(f"Direct children: {tree.child_count}")

# Find nodes by path
child = tree.find_by_path(0, 1)  # First child's second child
```

### Auto-Indexing

```python
# Nodes are automatically indexed
root = Tree(content=DefaultContent(name="Root"))
child1 = root.add_child(Leaf(content=DefaultContent(name="Child 1")))
child2 = root.add_child(Leaf(content=DefaultContent(name="Child 2")))

# Access auto-generated properties
print(child1.node_id)  # "0"
print(child2.node_id)  # "1"
print(child1.level)    # 1 (depth from root)
```

## Type Aliases

For convenience, the module provides type aliases for common patterns:

```python
from haive.core.common.structures import SimpleTree, SimpleLeaf

# SimpleTree = Tree with default types
tree = SimpleTree(content=DefaultContent(name="Simple"))
leaf = SimpleLeaf(content=DefaultContent(name="Simple Leaf"))
```

## Integration with Planning

The tree_leaf structure is designed to work seamlessly with the planning_v2 module:

```python
from haive.agents.planning_v2.base.models import Plan, Task
from haive.core.common.structures import Tree, Leaf

# Plans can use Tree structure
plan_tree = Tree[Task, Tree[Task, str], str](
    content=Task(name="Build Feature", description="...")
)

# Or be used as content in trees
meta_tree = Tree[Plan[Task], Tree[Plan[Task], str], str](
    content=Plan(objective="Q1 Goals", steps=[...])
)
```

## Advanced Features

### Heterogeneous Trees

```python
# Trees can contain mixed child types
from typing import Union

MixedChild = Union[Leaf[TaskContent, str], Tree[TaskContent, 'MixedChild', str]]

mixed_tree: Tree[TaskContent, MixedChild, str] = Tree(
    content=TaskContent(name="Mixed", priority=1)
)

# Add both leaves and subtrees to same parent
mixed_tree.add_child(Leaf(content=TaskContent(name="Leaf", priority=2)))
mixed_tree.add_child(Tree(content=TaskContent(name="Subtree", priority=3)))
```

### Custom Node Types

```python
from haive.core.common.structures import TreeNode

class CustomNode(TreeNode[MyContent, MyResult]):
    """Custom node with additional behavior."""

    custom_field: str = "default"

    def is_leaf(self) -> bool:
        return not hasattr(self, 'children')

    def custom_method(self) -> str:
        return f"Custom: {self.content.name}"
```

## Best Practices

1. **Use Type Hints**: Always specify generic parameters for type safety
2. **Leverage Computed Properties**: Use built-in properties instead of manual calculation
3. **Auto-Indexing**: Let the framework handle indexing automatically
4. **Consistent Types**: Keep content/result types consistent within a tree
5. **Default Types**: Use DefaultContent/DefaultResult for simple use cases

## Migration from AutoTree

If migrating from the older AutoTree:

```python
# Old AutoTree
from haive.core.common.structures.tree import AutoTree
old_tree = AutoTree(my_model)

# New tree_leaf
from haive.core.common.structures import Tree, auto_tree
new_tree = auto_tree(my_model)  # Auto conversion
# OR
new_tree = Tree(content=my_model)  # Manual creation
```

## Future Enhancements

- Full AutoTree functionality for automatic BaseModel conversion
- Serialization/deserialization support
- Visualization improvements
- Async node execution
- Tree diffing and merging
