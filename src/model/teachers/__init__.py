"""Teacher models for knowledge distillation."""

from src.model.teachers.bge_m3 import (
    BGEM3Teacher,
    create_bge_m3_teacher,
)

__all__ = [
    "BGEM3Teacher",
    "create_bge_m3_teacher",
]
