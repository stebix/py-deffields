"""Shared type definitions for jaxtyping annotations."""

from beartype import beartype
from jaxtyping import jaxtyped

# Decorator that enables runtime shape checking with beartype
jaxcheck = jaxtyped(typechecker=beartype)
