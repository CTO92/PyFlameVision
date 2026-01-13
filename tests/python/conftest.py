"""
Pytest configuration and fixtures for PyFlameVision tests.
"""

import pytest
import sys
from pathlib import Path

# Add Python package to path for testing
project_root = Path(__file__).parent.parent.parent
python_path = project_root / "python"
if str(python_path) not in sys.path:
    sys.path.insert(0, str(python_path))


@pytest.fixture
def sample_rgb_shape():
    """Standard RGB image shape [batch, channels, height, width]."""
    return [1, 3, 224, 224]


@pytest.fixture
def sample_batch_shape():
    """Batch of RGB images."""
    return [8, 3, 224, 224]


@pytest.fixture
def sample_grayscale_shape():
    """Grayscale image shape."""
    return [1, 1, 224, 224]


@pytest.fixture
def imagenet_mean():
    """ImageNet normalization mean values."""
    return [0.485, 0.456, 0.406]


@pytest.fixture
def imagenet_std():
    """ImageNet normalization standard deviation values."""
    return [0.229, 0.224, 0.225]
