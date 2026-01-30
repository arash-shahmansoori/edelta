"""
Data Preparation Modules

This module contains dataset preparation scripts for the benchmarks:
- gyroscope: Continuous rotation prediction (manifold precision test)
- stability: Norm preservation over long sequences (isometry test)
- reflection: Pure negation task y = -x (direct geometric operator test)
"""

from . import gyroscope
from . import stability
from . import reflection

__all__ = ['gyroscope', 'stability', 'reflection']
