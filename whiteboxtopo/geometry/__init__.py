"""Curvature, the Lipschitz field, the tubular neighbourhood, and where the model will fail."""
from .core import (second_fundamental_form, lipschitz_field, reach, ignorance,
                   failure_risk, confidence_saddles, manifold_margin, tubular_distance)
__all__ = ['second_fundamental_form', 'lipschitz_field', 'reach', 'ignorance',
           'failure_risk', 'confidence_saddles', 'manifold_margin', 'tubular_distance']
