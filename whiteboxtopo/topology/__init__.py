"""Cohomology of the representation manifold and the coordinates it defines."""
from .equivariant import induced_shift_operator, isotypic_planes, isotypic_decomposition
from .core import (persistence, significant_classes, normalized_persistence,
                   circular_coordinates, circular_correlation, circular_rank_correlation, circular_linear_correlation,
                   name_coordinate, toroidal_decorrelate, fiber_topology, match_classes)
__all__ = ['induced_shift_operator', 'isotypic_planes', 'isotypic_decomposition', 'persistence', 'significant_classes', 'normalized_persistence', 'circular_coordinates',
           'circular_correlation', 'circular_rank_correlation', 'circular_linear_correlation', 'name_coordinate',
           'toroidal_decorrelate', 'fiber_topology', 'match_classes']
