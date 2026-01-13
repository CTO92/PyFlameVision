# Neural Network (nn) Module API Reference

The `pyflame_vision.nn` module provides PyTorch-compatible neural network layer abstractions for building model architectures.

## Core Classes

### TensorSpec

Tensor specification for shape inference.

```python
from pyflame_vision.nn import TensorSpec

spec = TensorSpec(shape=[1, 3, 224, 224], dtype="float32")
print(spec.shape)      # [1, 3, 224, 224]
print(spec.dtype)      # "float32"
print(spec.numel())    # 150528
print(spec.size_bytes())  # 602112
```

**Attributes:**
- `shape: List[int]` - Tensor dimensions
- `dtype: str` - Data type ("float32", "float16", etc.)

**Methods:**
- `numel() -> int` - Total number of elements
- `size_bytes() -> int` - Size in bytes

### Parameter

Parameter metadata for weight loading.

```python
from pyflame_vision.nn import Parameter

# Parameters are returned from modules
params = model.parameters()
for p in params:
    print(f"{p.name}: {p.spec.shape}, requires_grad={p.requires_grad}")
```

**Attributes:**
- `name: str` - Parameter name
- `spec: TensorSpec` - Shape and dtype information
- `requires_grad: bool` - Whether parameter is learnable

### Module

Base class for all neural network layers.

```python
from pyflame_vision.nn import Module

# Module is abstract - use derived classes
module.forward(input_spec)         # Compute output shape
module.get_output_shape(shape)     # Convenience method
module.name()                      # Get layer name
module.parameters()                # Get parameters
module.named_parameters()          # Get named parameters
module.train(mode=True)            # Set training mode
module.eval()                      # Set evaluation mode
module.is_training()               # Check mode
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `forward(input: TensorSpec)` | `TensorSpec` | Compute output shape |
| `get_output_shape(shape: List[int])` | `List[int]` | Get output shape for input |
| `name()` | `str` | Layer type name |
| `parameters()` | `List[Parameter]` | All parameters |
| `named_parameters(prefix="")` | `Dict[str, Parameter]` | Hierarchical parameters |
| `train(mode=True)` | `None` | Set training mode |
| `eval()` | `None` | Set evaluation mode |
| `is_training()` | `bool` | Check if in training mode |

---

## Containers

### Sequential

Container that chains modules together.

```python
from pyflame_vision.nn import Sequential, Conv2d, BatchNorm2d, ReLU

# Create from list
seq = Sequential([
    Conv2d(3, 64, kernel_size=3, padding=1),
    BatchNorm2d(64),
    ReLU()
])

# Or build incrementally
seq = Sequential()
seq.add(Conv2d(3, 64, kernel_size=3, padding=1))
seq.add(BatchNorm2d(64))
seq.add(ReLU())

# Usage
output = seq.get_output_shape([1, 3, 224, 224])  # [1, 64, 224, 224]

# Access modules
print(len(seq))      # 3
first = seq[0]       # Conv2d
print(seq.empty())   # False
```

**Constructor:**
```python
Sequential(modules: List[Module] = [])
```

**Methods:**
- `add(module: Module)` - Add module to end
- `__len__()` - Number of modules
- `__getitem__(index)` - Get module by index
- `empty()` - Check if empty

---

## Convolution Layers

### Conv2d

2D convolution layer.

```python
from pyflame_vision.nn import Conv2d

# Basic usage
conv = Conv2d(in_channels=3, out_channels=64, kernel_size=3)

# With padding (same output size)
conv = Conv2d(3, 64, kernel_size=3, padding=1)

# Strided convolution (downsampling)
conv = Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

# Grouped convolution
conv = Conv2d(64, 64, kernel_size=3, groups=64)  # Depthwise

# Without bias
conv = Conv2d(3, 64, kernel_size=3, bias=False)

# Shape inference
output = conv.get_output_shape([1, 3, 224, 224])
```

**Constructor:**
```python
Conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: PaddingMode = PaddingMode.ZEROS
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_channels` | int | required | Input channels |
| `out_channels` | int | required | Output channels |
| `kernel_size` | int | required | Convolution kernel size |
| `stride` | int | 1 | Stride |
| `padding` | int | 0 | Zero-padding |
| `dilation` | int | 1 | Kernel dilation |
| `groups` | int | 1 | Grouped convolution |
| `bias` | bool | True | Include bias term |
| `padding_mode` | PaddingMode | ZEROS | Padding mode |

**Properties:**
- `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`, `dilation`, `groups`

**Methods:**
- `has_bias() -> bool` - Check if bias is enabled
- `is_depthwise() -> bool` - Check if depthwise convolution
- `is_pointwise() -> bool` - Check if 1x1 convolution
- `halo_size() -> int` - Get halo size for distributed execution

**Output Shape Formula:**
```
H_out = floor((H_in + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
W_out = floor((W_in + 2*padding - dilation*(kernel-1) - 1) / stride + 1)
```

### PaddingMode

Enum for convolution padding modes.

```python
from pyflame_vision.nn import PaddingMode

conv = Conv2d(3, 64, 3, padding=1, padding_mode=PaddingMode.REFLECT)
```

| Value | Description |
|-------|-------------|
| `ZEROS` | Zero padding (default) |
| `REFLECT` | Reflect padding |
| `REPLICATE` | Replicate border |
| `CIRCULAR` | Circular/wrap padding |

---

## Normalization Layers

### BatchNorm2d

2D batch normalization.

```python
from pyflame_vision.nn import BatchNorm2d

bn = BatchNorm2d(num_features=64)

# With custom parameters
bn = BatchNorm2d(
    num_features=64,
    eps=1e-5,
    momentum=0.1,
    affine=True,
    track_running_stats=True
)

# Shape unchanged
output = bn.get_output_shape([1, 64, 56, 56])  # [1, 64, 56, 56]
```

**Constructor:**
```python
BatchNorm2d(
    num_features: int,
    eps: float = 1e-5,
    momentum: float = 0.1,
    affine: bool = True,
    track_running_stats: bool = True
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_features` | int | required | Number of channels (C in NCHW) |
| `eps` | float | 1e-5 | Numerical stability constant |
| `momentum` | float | 0.1 | Running stats momentum |
| `affine` | bool | True | Learnable affine parameters |
| `track_running_stats` | bool | True | Track running mean/var |

---

## Activation Functions

### ReLU

Rectified Linear Unit: `max(0, x)`

```python
from pyflame_vision.nn import ReLU

relu = ReLU()
relu = ReLU(inplace=True)  # In-place hint

output = relu.get_output_shape([1, 64, 56, 56])  # [1, 64, 56, 56]
```

### SiLU (Swish)

Sigmoid Linear Unit: `x * sigmoid(x)`

```python
from pyflame_vision.nn import SiLU

silu = SiLU()
silu = SiLU(inplace=True)
```

### Sigmoid

Sigmoid activation: `1 / (1 + exp(-x))`

```python
from pyflame_vision.nn import Sigmoid

sigmoid = Sigmoid()
```

### Identity

Pass-through layer (no-op).

```python
from pyflame_vision.nn import Identity

identity = Identity()
output = identity.get_output_shape([1, 64, 56, 56])  # [1, 64, 56, 56]
```

---

## Pooling Layers

### MaxPool2d

2D max pooling.

```python
from pyflame_vision.nn import MaxPool2d

# Basic 2x2 pooling
pool = MaxPool2d(kernel_size=2, stride=2)

# With padding
pool = MaxPool2d(kernel_size=3, stride=2, padding=1)

# Full parameters
pool = MaxPool2d(
    kernel_size=3,
    stride=2,
    padding=1,
    dilation=1,
    ceil_mode=False
)

output = pool.get_output_shape([1, 64, 224, 224])  # [1, 64, 112, 112]
```

**Constructor:**
```python
MaxPool2d(
    kernel_size: int,
    stride: int = 0,  # 0 means same as kernel_size
    padding: int = 0,
    dilation: int = 1,
    ceil_mode: bool = False
)
```

### AvgPool2d

2D average pooling.

```python
from pyflame_vision.nn import AvgPool2d

pool = AvgPool2d(kernel_size=2, stride=2)

pool = AvgPool2d(
    kernel_size=3,
    stride=2,
    padding=1,
    ceil_mode=False,
    count_include_pad=True
)
```

### AdaptiveAvgPool2d

Adaptive average pooling (outputs fixed size).

```python
from pyflame_vision.nn import AdaptiveAvgPool2d

# Output 1x1 (global average pooling)
pool = AdaptiveAvgPool2d(output_size=1)
output = pool.get_output_shape([1, 2048, 7, 7])  # [1, 2048, 1, 1]

# Output 7x7
pool = AdaptiveAvgPool2d(output_size=7)
output = pool.get_output_shape([1, 512, 14, 14])  # [1, 512, 7, 7]
```

---

## Linear Layers

### Linear

Fully connected (dense) layer.

```python
from pyflame_vision.nn import Linear

fc = Linear(in_features=2048, out_features=1000)

# Without bias
fc = Linear(2048, 1000, bias=False)

output = fc.get_output_shape([1, 2048])  # [1, 1000]
```

**Constructor:**
```python
Linear(
    in_features: int,
    out_features: int,
    bias: bool = True
)
```

### Flatten

Flatten dimensions.

```python
from pyflame_vision.nn import Flatten

# Flatten all except batch (default)
flatten = Flatten()
output = flatten.get_output_shape([1, 64, 7, 7])  # [1, 3136]

# Custom start/end dimensions
flatten = Flatten(start_dim=1, end_dim=-1)
```

**Constructor:**
```python
Flatten(
    start_dim: int = 1,
    end_dim: int = -1
)
```

---

## Weight Parameters

Each layer type has specific parameter shapes:

### Conv2d Parameters
```
weight: [out_channels, in_channels/groups, kernel_h, kernel_w]
bias: [out_channels] (if bias=True)
```

### BatchNorm2d Parameters
```
weight (gamma): [num_features]
bias (beta): [num_features]
running_mean: [num_features] (not learnable)
running_var: [num_features] (not learnable)
num_batches_tracked: [1] (not learnable)
```

### Linear Parameters
```
weight: [out_features, in_features]
bias: [out_features] (if bias=True)
```

---

## Thread Safety

All nn modules are thread-safe after construction:
- Modules are immutable (parameters are metadata only)
- Multiple threads can call `forward()` concurrently
- No internal state is modified during shape inference
