"""
Training Scripts

- train_continuous: Training for continuous physics benchmarks (gyroscope, correction, stability)
- train_language_model: Original GPT language model training script
"""

from .train_continuous import (
    ContinuousModelWrapper,
    create_model,
    load_dataset,
    train,
)

__all__ = [
    'ContinuousModelWrapper',
    'create_model', 
    'load_dataset',
    'train',
]
