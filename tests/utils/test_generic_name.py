"""Test what happens with generic class names."""

from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")

class Plan(BaseModel, Generic[T]):
    """Generic plan."""

class Task(BaseModel):
    """Task model."""

# Test generic name
plan_task = Plan[Task]
print(f"Plan[Task].__name__ = {getattr(plan_task, '__name__', 'NO __name__ ATTRIBUTE')}")
print(f"Plan[Task].__class__.__name__ = {plan_task.__class__.__name__}")
print(f"str(Plan[Task]) = {plan_task!s}")

# Test concrete class
class TaskPlan(Plan[Task]):
    """Concrete plan."""

print(f"\nTaskPlan.__name__ = {TaskPlan.__name__}")
print(f"str(TaskPlan) = {TaskPlan!s}")
