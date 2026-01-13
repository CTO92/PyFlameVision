# PyFlameVision Architecture

**Version:** 6.0
**Date:** January 12, 2026

---

## 1. Executive Summary

PyFlameVision is a standalone computer vision library designed for the Cerebras Wafer-Scale Engine (WSE). It provides PyTorch-compatible APIs for image transforms, datasets, and vision operations. PyFlameVision can operate independently as a pure Python library or integrate with PyFlame for hardware-accelerated execution on Cerebras hardware.

### Repository Structure

PyFlameVision is maintained as an **independent repository**, following the PyTorch/torchvision pattern:

```
PyFlameVision (this repository)
├── Standalone Python functionality (transforms, datasets, io, ops)
├── C++ core with CSL templates for Cerebras optimization
└── Optional dependency on PyFlame for hardware acceleration

PyFlame (separate repository)
├── Core tensor library
├── IR/Graph system
└── Cerebras SDK integration
```

**Rationale for Separation:**
- **Independent versioning** - Vision library can release faster than core
- **Smaller dependencies** - Users can install only what they need
- **Community contribution** - Lower barrier for vision-specific contributions
- **Industry standard** - Follows PyTorch + torchvision, TensorFlow + tf.image patterns

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Lazy Evaluation** | Follows PyFlame's model; enables whole-graph optimization before CSL compilation |
| **torchvision API Compatibility** | Minimizes migration effort for existing PyTorch vision code |
| **NCHW Image Format** | Standard deep learning format; efficient PE mesh mapping |
| **Template-Based CSL Generation** | Maintainable, debuggable, extensible kernel generation |
| **Halo Exchange Pattern** | Enables distributed image processing with neighbor data sharing |
| **Module Abstraction** | PyTorch-compatible nn.Module pattern for model architectures |
| **Immutable Modules** | Thread-safe model inference through immutable module design |
| **Thread-Safe RNG** | Mutex-protected random transforms for concurrent training |
| **Secure Seeding** | Cryptographically robust seed generation with reproducibility support |
| **Detection Ops** | GridSample, ROIAlign, NMS for object detection pipelines |
| **Security Limits** | Validated bounds for ROI counts, grid coordinates, and NMS boxes |
| **Dataset Abstraction** | PyTorch-compatible Dataset/DataLoader for seamless data pipelines |
| **Multiprocess Loading** | Thread-based workers for parallel data loading with prefetching |
| **Standalone Mode** | Full functionality without PyFlame dependency |
| **Optional PyFlame** | Hardware acceleration available when PyFlame is installed |

### Installation Modes

PyFlameVision supports two installation modes:

**1. Pure Python (Standalone)**
```bash
pip install pyflame-vision
```
- Full Python API (transforms, datasets, io, ops)
- NumPy-based operations
- No hardware acceleration
- No C++ compilation required

**2. With PyFlame Integration**
```bash
pip install "pyflame-vision[pyflame]"
# or
export PYFLAME_DIR=/path/to/pyflame
pip install pyflame-vision
```
- All standalone features
- Hardware-accelerated operations on Cerebras WSE
- C++ extension compiled against PyFlame
- Lazy evaluation with graph optimization

---

## 2. System Architecture

### 2.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Application                                │
│   import pyflame_vision.transforms as T                                     │
│   import pyflame_vision.datasets as D                                       │
│   dataset = D.ImageFolder("data/train", transform=T.Resize(224))           │
│   loader = D.DataLoader(dataset, batch_size=32, shuffle=True)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PyFlameVision Python API                           │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  transforms/           nn/                  models/                   │  │
│  │  ├── Resize           ├── Module           ├── ResNet                │  │
│  │  ├── CenterCrop       ├── Sequential       ├── EfficientNet          │  │
│  │  ├── RandomCrop       ├── Conv2d           ├── resnet18/34/50/101    │  │
│  │  ├── Normalize        ├── BatchNorm2d      └── efficientnet_b0-b4    │  │
│  │  ├── Compose          ├── ReLU/SiLU                                  │  │
│  │  │                    ├── MaxPool2d        functional/                │  │
│  │  │ Phase 3 Augments:  ├── Linear           ├── resize()              │  │
│  │  ├── RandomHFlip      └── Flatten          ├── crop()                │  │
│  │  ├── RandomVFlip                           └── normalize()           │  │
│  │  ├── RandomRotation   io/                 ops/ (Phase 4)             │  │
│  │  ├── ColorJitter      ├── read_image      ├── GridSample             │  │
│  │  └── GaussianBlur     ├── write_image     ├── ROIAlign               │  │
│  │                       ├── decode_image    ├── NMS/BatchedNMS         │  │
│  │  datasets/ (Phase 5)  └── encode_image    └── SoftNMS                │  │
│  │  ├── Dataset                                                          │  │
│  │  ├── IterableDataset                                                  │  │
│  │  ├── ImageFolder                                                      │  │
│  │  ├── DataLoader                                                       │  │
│  │  └── Samplers                                                         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ pybind11
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PyFlameVision C++ Core                              │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────────────────┐  │
│  │  core/         │  │  transforms/   │  │  nn/                        │  │
│  │  ├─ImageTensor │  │  ├─Resize      │  │  ├─Module (base class)      │  │
│  │  ├─Interpolation│ │  ├─CenterCrop  │  │  ├─Sequential/ModuleList    │  │
│  │  ├─ColorSpace  │  │  ├─RandomCrop  │  │  ├─Conv2d, BatchNorm2d      │  │
│  │  └─Security    │  │  ├─Normalize   │  │  ├─ReLU, SiLU, Sigmoid      │  │
│  │                │  │  └─Compose     │  │  ├─MaxPool2d, AvgPool2d     │  │
│  │                │  │                │  │  └─Linear, Flatten          │  │
│  └────────────────┘  └────────────────┘  └─────────────────────────────┘  │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────────────────┐  │
│  │  models/       │  │  ops/          │  │  backend/                   │  │
│  │  ├─ResNet      │  │  ├─GridSample  │  │  ├─VisionCSLCodeGenerator   │  │
│  │  ├─EfficientNet│  │  ├─ROIAlign    │  │  ├─VisionCSLTemplates       │  │
│  │  ├─BasicBlock  │  │  ├─NMS         │  │  └─InterpolationKernels     │  │
│  │  ├─Bottleneck  │  │  ├─BatchedNMS  │  │                             │  │
│  │  └─MBConv      │  │  └─SoftNMS     │  │                             │  │
│  └────────────────┘  └────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ Links against
┌─────────────────────────────────────────────────────────────────────────────┐
│                             PyFlame Core Library                            │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────────────────┐  │
│  │  core/         │  │  ir/           │  │  backend/                   │  │
│  │  ├─Tensor      │  │  ├─Graph       │  │  ├─CSLCodeGenerator         │  │
│  │  ├─DType       │  │  ├─Node        │  │  ├─CSLTemplates             │  │
│  │  └─Layout      │  │  └─Operation   │  │  └─CSLCompiler              │  │
│  │                │  │                │  │                             │  │
│  │                │  │                │  │  runtime/                   │  │
│  │                │  │                │  │  └─Executor                 │  │
│  └────────────────┘  └────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ Generates CSL
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Cerebras SDK (cslc)                               │
│               Compiles CSL to WSE binary for execution                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ Executes on
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Cerebras WSE Hardware                                │
│                  850,000+ PEs with 2D mesh interconnect                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
User Code                  PyFlameVision               PyFlame                WSE
   │                            │                         │                    │
   │  transform(image)          │                         │                    │
   ├───────────────────────────►│                         │                    │
   │                            │                         │                    │
   │                            │  Create IR nodes        │                    │
   │                            ├────────────────────────►│                    │
   │                            │                         │                    │
   │                            │  VISION_RESIZE node     │                    │
   │                            │  VISION_NORMALIZE node  │                    │
   │                            │◄────────────────────────┤                    │
   │                            │                         │                    │
   │  .eval()                   │                         │                    │
   ├───────────────────────────────────────────────────►  │                    │
   │                            │                         │                    │
   │                            │  Graph optimization     │                    │
   │                            │  ├─Fusion               │                    │
   │                            │  ├─Layout selection     │                    │
   │                            │  └─Memory planning      │                    │
   │                            │                         │                    │
   │                            │  CSL code generation    │                    │
   │                            │  ├─Vision templates     │                    │
   │                            │  └─Halo exchange        │                    │
   │                            │                         │                    │
   │                            │                         │  Compile CSL       │
   │                            │                         ├───────────────────►│
   │                            │                         │                    │
   │                            │                         │  Execute           │
   │                            │                         │◄───────────────────┤
   │                            │                         │                    │
   │◄──────────────────────────────────────────────────────────────────────────┤
   │  Result tensor                                                            │
```

---

## 3. Component Design

### 3.1 Core Module

The core module provides vision-specific utilities that extend PyFlame's tensor system.

```
core/
├── ImageTensor       - NCHW tensor utilities, dimension access
├── InterpolationMode - Interpolation algorithms (nearest, bilinear, bicubic)
└── ColorSpace        - Color space definitions (RGB, BGR, GRAY, HSV)
```

**Key Responsibilities:**
- Validate tensors are valid images (4D NCHW format)
- Compute optimal PE mesh layouts for image dimensions
- Calculate tile shapes and halo sizes for distributed processing
- Provide dimension accessors (batch, channels, height, width)

**Design Pattern:** Static utility class with helper methods

### 3.2 Transforms Module

Transforms are the primary user-facing API, mirroring torchvision's design.

```
transforms/
├── Transform (base)     - Abstract base class
├── SizeTransform        - Base for size-based transforms
├── Resize               - Image resizing with interpolation
├── CenterCrop           - Deterministic center cropping
├── RandomCrop           - Stochastic random cropping
├── Normalize            - Channel-wise normalization
├── Compose              - Transform pipeline container
└── functional/          - Stateless functional API
    ├── resize()
    ├── crop()
    ├── center_crop()
    └── normalize()
```

**Design Patterns:**
- **Strategy Pattern:** Interpolation modes as strategy
- **Composite Pattern:** Compose aggregates transforms
- **Decorator Pattern:** Transforms wrap input tensors

**Key Design Decisions:**

1. **Lazy Execution:** All transforms create IR graph nodes rather than executing immediately
2. **Immutability:** Transforms don't modify input tensors; they create new output tensors
3. **Reproducibility:** Random transforms support seed setting for determinism

### 3.3 Backend Module

The backend extends PyFlame's CSL code generation with vision-specific kernels.

```
backend/
├── VisionCSLCodeGenerator  - Extended code generator
├── VisionCSLTemplates      - Template registry for vision ops
├── InterpolationKernels    - Interpolation algorithm implementations
└── csl_templates/
    ├── resize_bilinear.csl.template
    ├── resize_bicubic.csl.template
    ├── crop.csl.template
    ├── normalize.csl.template
    ├── halo_exchange.csl.template
    └── ops/
        ├── grid_sample.csl.template  - Coordinate-based sampling
        ├── roi_align.csl.template    - ROI feature extraction
        └── nms.csl.template          - Non-maximum suppression
```

**Key Responsibilities:**
- Generate efficient CSL kernels for vision operations
- Handle halo exchange for operations requiring neighbor data
- Optimize memory access patterns for 2D image data
- Integrate with PyFlame's compilation pipeline

### 3.4 Neural Network (nn) Module

The nn module provides PyTorch-compatible neural network layer abstractions.

```
nn/
├── module.hpp          - Base Module class, TensorSpec, Parameter
├── container.hpp       - Sequential, ModuleList containers
├── activation.hpp      - ReLU, SiLU, Sigmoid, GELU, Identity, Flatten
├── conv.hpp            - Conv2d with grouped/depthwise support
├── batchnorm.hpp       - BatchNorm2d, BatchNorm1d, LayerNorm
├── pooling.hpp         - MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
└── linear.hpp          - Linear (fully connected) layer
```

**Key Design Decisions:**

1. **TensorSpec for Shape Inference:** Modules compute output shapes via `forward(TensorSpec)` without actual data
2. **Immutable Modules:** All modules are thread-safe after construction
3. **Parameter Metadata:** Parameters store shape/dtype info for weight loading compatibility
4. **PyTorch API Compatibility:** Constructor signatures match PyTorch's nn module

**Design Patterns:**
- **Composite Pattern:** Sequential aggregates modules
- **Template Method:** Base Module defines forward() interface
- **Strategy Pattern:** Activation functions as interchangeable modules

### 3.5 Models Module

The models module provides complete model architectures optimized for Cerebras.

```
models/
├── resnet.hpp          - ResNet family (BasicBlock, Bottleneck, ResNet)
├── efficientnet.hpp    - EfficientNet family (SE, MBConv, EfficientNet)
└── models.hpp          - Main header aggregating all models
```

**Supported Models:**

| Family | Variants | Key Features |
|--------|----------|--------------|
| **ResNet** | 18, 34, 50, 101, 152 | BasicBlock/Bottleneck, ResNeXt, WideResNet |
| **EfficientNet** | B0-B4 | MBConv blocks, Squeeze-Excitation, compound scaling |

**Model Architecture Pattern:**

```cpp
class ResNet : public nn::Module {
public:
    // Shape inference
    TensorSpec forward(const TensorSpec& input) const override;

    // Feature extraction (returns intermediate shapes)
    std::vector<TensorSpec> forward_features(const TensorSpec& input) const;

    // Modify for transfer learning
    void remove_fc();  // Remove classifier for feature extraction

    // Access internal structure
    const Sequential& layer1() const;
    // ...
};
```

### 3.6 Operations (ops) Module

The ops module provides specialized computer vision operations commonly used in detection and segmentation models.

```
ops/
├── grid_sample.hpp     - Coordinate-based spatial sampling
├── roi_align.hpp       - Region of Interest feature extraction
├── nms.hpp             - Non-Maximum Suppression variants
└── ops.hpp             - Aggregate header
```

**Key Operations:**

| Operation | Description | Use Case |
|-----------|-------------|----------|
| **GridSample** | Sample from normalized [-1,1] grid coordinates | Spatial transformers, deformable convolutions |
| **ROIAlign** | Extract fixed-size features from regions | Faster R-CNN, Mask R-CNN detection heads |
| **NMS** | Filter overlapping detection boxes | Object detection post-processing |
| **BatchedNMS** | Class-aware NMS per batch item | Multi-class detection |
| **SoftNMS** | Score decay instead of hard suppression | Improved recall in dense scenes |

**Design Patterns:**
- **Shape Inference:** All ops provide `get_output_shape()` for lazy evaluation
- **Security Validation:** Bounds checking on ROI counts, grid coordinates, NMS boxes
- **torchvision Compatibility:** API matches `torchvision.ops` signatures

**Security Limits:**
```cpp
struct SecurityLimits {
    static constexpr int64_t MAX_ROIS = 10000;
    static constexpr int64_t MAX_ROI_OUTPUT_SIZE = 256;
    static constexpr int64_t MAX_GRID_SAMPLE_SIZE = 4096;
    static constexpr int64_t MAX_NMS_BOXES = 100000;
    static constexpr float MAX_GRID_COORDINATE = 1e6f;
};
```

### 3.7 Datasets Module

The datasets module provides PyTorch-compatible data loading utilities.

```
datasets/
├── dataset.py          - Dataset, IterableDataset, ConcatDataset, Subset
├── vision_dataset.py   - VisionDataset base class with transforms
├── image_folder.py     - ImageFolder, DatasetFolder
├── samplers.py         - All sampler implementations
├── collate.py          - Collation functions
└── dataloader.py       - DataLoader with multiprocessing
```

**Key Components:**

| Component | Description |
|-----------|-------------|
| **Dataset** | Abstract base class for map-style datasets (random access) |
| **IterableDataset** | Abstract base class for streaming datasets (sequential) |
| **VisionDataset** | Base class with transform/target_transform support |
| **ImageFolder** | Directory-based loading where subdirs are classes |
| **DataLoader** | Batched iteration with optional worker threads |
| **Samplers** | Index generation strategies for data access order |

**Design Patterns:**
- **Strategy Pattern:** Samplers as interchangeable index strategies
- **Template Method:** Dataset base defines __getitem__/__len__ interface
- **Composite Pattern:** ConcatDataset aggregates multiple datasets
- **Iterator Pattern:** DataLoader provides iteration over batches

**Security Features:**
```python
class DatasetSecurityLimits:
    MAX_PATH_LENGTH = 4096
    MAX_DIRECTORY_DEPTH = 100
    MAX_DATASET_SIZE = 1 << 30  # ~1 billion

class DataLoaderSecurityLimits:
    MAX_NUM_WORKERS = 64
    MAX_BATCH_SIZE = 65536
    MAX_PREFETCH_FACTOR = 100
```

### 3.8 IO Module

The io module provides image loading and saving utilities.

```
io/
├── __init__.py         - Module exports
└── io.py               - Image I/O functions
```

**Key Functions:**

| Function | Description |
|----------|-------------|
| **read_image** | Load image file to [C, H, W] numpy array |
| **write_image** | Save [C, H, W] array to image file |
| **decode_image** | Decode image bytes to array |
| **encode_image** | Encode array to image bytes |

**Image Read Modes:**
- `UNCHANGED` - Keep original format
- `GRAY` - Convert to grayscale
- `RGB` - Convert to 3-channel RGB
- `RGB_ALPHA` - RGB with alpha channel

**Security Features:**
```python
class IOSecurityLimits:
    MAX_IMAGE_WIDTH = 65536
    MAX_IMAGE_HEIGHT = 65536
    MAX_FILE_SIZE = 1 << 30  # 1 GB
    MAX_PATH_LENGTH = 4096
```

---

## 4. Memory and Layout Strategy

### 4.1 Image Tiling for PE Mesh

Images are distributed across the PE mesh by tiling:

```
Input Image [1, 3, 224, 224]
           │
           ▼
┌─────────────────────────────────────────────────┐
│  Layout: Grid(4, 4) = 16 PEs                    │
│                                                 │
│  ┌────────┬────────┬────────┬────────┐         │
│  │ PE(0,0)│ PE(0,1)│ PE(0,2)│ PE(0,3)│         │
│  │ 56x56  │ 56x56  │ 56x56  │ 56x56  │         │
│  ├────────┼────────┼────────┼────────┤         │
│  │ PE(1,0)│ PE(1,1)│ PE(1,2)│ PE(1,3)│         │
│  │ 56x56  │ 56x56  │ 56x56  │ 56x56  │         │
│  ├────────┼────────┼────────┼────────┤         │
│  │  ...   │  ...   │  ...   │  ...   │         │
│  └────────┴────────┴────────┴────────┘         │
│                                                 │
│  Each PE stores: 3 channels × 56×56 = 9,408 f32│
│  Memory per PE: 37.6 KB (fits in 48KB SRAM)    │
└─────────────────────────────────────────────────┘
```

### 4.2 Halo Exchange Pattern

Operations like resize and blur require neighboring pixel data:

```
┌─────────────────────────────────────────────────────────────┐
│  Halo Exchange for Bilinear Interpolation (halo=1)          │
│                                                             │
│  PE(1,1) tile with halo:                                    │
│  ┌───────────────────────────────────────┐                  │
│  │  From PE(0,0) │  From PE(0,1)         │  ← North halo    │
│  ├───────────────┼───────────────────────┤                  │
│  │  From         │                       │                  │
│  │  PE(1,0)      │  Local data          │                  │
│  │               │  (interior)           │                  │
│  │  ← West       │                       │                  │
│  │    halo       │                       │                  │
│  ├───────────────┼───────────────────────┤                  │
│  │  From PE(2,0) │  From PE(2,1)         │  ← South halo    │
│  └───────────────┴───────────────────────┘                  │
│                                                             │
│  Wavelet Communication:                                     │
│  1. Pack border pixels into send buffers                    │
│  2. Send to neighbors (N, S, E, W)                          │
│  3. Receive from neighbors into halo regions                │
│  4. Execute kernel with complete neighborhood               │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Layout Selection Algorithm

```
optimal_layout(height, width, element_size):
    image_bytes = height × width × element_size

    if image_bytes ≤ 32KB:
        return SinglePE()

    # Target 16KB tiles (leave room for halos and scratch)
    target_tile_bytes = 16KB
    target_tile_elements = target_tile_bytes / element_size
    total_elements = height × width
    num_tiles = ceil(total_elements / target_tile_elements)

    # Find grid dimensions
    grid_rows = sqrt(num_tiles)
    grid_cols = ceil(num_tiles / grid_rows)

    # Adjust for image aspect ratio
    aspect = height / width
    if aspect > 1.5:  # Tall image
        grid_rows = min(grid_rows × 2, height)
        grid_cols = ceil(num_tiles / grid_rows)
    elif aspect < 0.67:  # Wide image
        grid_cols = min(grid_cols × 2, width)
        grid_rows = ceil(num_tiles / grid_cols)

    return Grid(grid_rows, grid_cols)
```

---

## 5. Operation Implementation Patterns

### 5.1 Transform Implementation Pattern

All transforms follow a consistent pattern:

```cpp
class ExampleTransform : public Transform {
public:
    ExampleTransform(params);

    pyflame::Tensor operator()(const pyflame::Tensor& input) const override {
        // 1. Validate input
        validate_image(input);

        // 2. Compute output specification
        auto output_spec = compute_output_spec(input);

        // 3. Get graph from input tensor
        auto& graph = input.impl()->graph();

        // 4. Create operation node with parameters
        auto op_node = graph->create_op(
            op_type(),
            {input.impl()->node()},
            output_spec,
            "example_" + std::to_string(graph->num_nodes())
        );

        // 5. Store parameters as metadata
        op_node->set_metadata("params", ...);

        // 6. Return new tensor referencing the operation node
        return pyflame::Tensor(
            pyflame::TensorImpl::from_node(graph, op_node)
        );
    }

private:
    // Transform parameters
};
```

### 5.2 CSL Template Pattern

CSL templates use parameter substitution:

```csl
// Template placeholders: {{PARAM_NAME}}

param memcpy = @import_module("<memcpy/get_params>", .{
    .width = {{PE_COLS}},
    .height = {{PE_ROWS}},
});

const TILE_HEIGHT: u32 = {{TILE_H}};
const TILE_WIDTH: u32 = {{TILE_W}};

var data: [{{NUM_CHANNELS}}][TILE_HEIGHT][TILE_WIDTH]{{DTYPE}};

task operation() void {
    // Generated operation code
    {{OPERATION_BODY}}
}
```

### 5.3 Halo-Aware Operation Pattern

Operations requiring neighbor data follow this pattern:

```
1. Allocate tile + halo buffer
   └── tile: [C][H][W]
   └── with_halo: [C][H + 2*halo][W + 2*halo]

2. Pack border data
   └── Extract top/bottom rows → send_north, send_south
   └── Extract left/right cols → send_west, send_east

3. Exchange halos (wavelet communication)
   └── Send to neighbors
   └── Receive from neighbors

4. Barrier synchronization

5. Execute kernel
   └── Access halo regions for boundary pixels
   └── Write results to interior only

6. Output tile (interior only)
```

---

## 6. Integration Points

### 6.1 Standalone vs Integrated Mode

PyFlameVision operates in two modes depending on PyFlame availability:

```python
# Automatic detection at import
import pyflame_vision

if pyflame_vision.HAS_PYFLAME:
    # PyFlame tensors, lazy evaluation, hardware acceleration
    from pyflame_vision._pyflame_vision_cpp import Tensor
else:
    # NumPy arrays, eager evaluation, CPU-only
    import numpy as np
```

### 6.2 PyFlame Integration (Optional)

When PyFlame is available, PyFlameVision provides deep integration:

| Integration Point | Description |
|-------------------|-------------|
| **Tensor** | Vision transforms consume and produce PyFlame Tensors |
| **Graph** | Transforms create nodes in PyFlame's IR Graph |
| **OpType** | Vision operations extend PyFlame's OpType enum (1000-1099) |
| **CSLCodeGenerator** | VisionCSLCodeGenerator inherits from PyFlame's generator |
| **Layout** | Uses PyFlame's MeshLayout for PE distribution |
| **Executor** | Compiled vision kernels run through PyFlame's executor |

### 6.3 Extension Mechanism

PyFlameVision extends PyFlame without modifying it:

```cpp
// PyFlame's OpType enum ends at CUSTOM = 999
// Vision ops start at 1000

// Registration at library initialization
void pyflame_vision_init() {
    auto& codegen = pyflame::backend::CSLCodeGenerator::instance();

    // Register vision operation handlers
    codegen.register_op_handler(
        OpType::VISION_RESIZE,
        [](const Operation& op) { return VisionCSLCodeGenerator::gen_resize(op); }
    );

    // ... register other vision ops
}
```

### 6.4 Standalone Fallback

Without PyFlame, operations use NumPy/PIL:

```python
# transforms/functional.py
def resize(img, size, interpolation=InterpolationMode.BILINEAR):
    if HAS_PYFLAME:
        # Create lazy IR node for later compilation
        return _pyflame_vision_cpp.resize(img, size, interpolation)
    else:
        # Eager NumPy/PIL execution
        from PIL import Image
        pil_img = Image.fromarray(img.transpose(1, 2, 0))
        pil_img = pil_img.resize(size, _pil_interpolation[interpolation])
        return np.array(pil_img).transpose(2, 0, 1)
```

---

## 7. Execution Flow

### 7.1 Complete Execution Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│ 1. USER CODE                                                               │
│    transform = T.Compose([T.Resize(224), T.CenterCrop(224), T.Normalize()])│
│    output = transform(image)                                               │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 2. GRAPH BUILDING (Lazy)                                                   │
│    Each transform creates an IR node:                                      │
│    image → VISION_RESIZE → VISION_CROP → VISION_NORMALIZE → output        │
│                                                                            │
│    No computation happens yet!                                             │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 3. EVALUATION TRIGGER                                                      │
│    output.eval()  or  print(output)  or  output.numpy()                   │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 4. OPTIMIZATION PASSES                                                     │
│    ├── Constant folding (fold normalize mean/std)                         │
│    ├── Operator fusion (fuse sequential transforms)                       │
│    ├── Layout optimization (select PE grid for each tensor)               │
│    └── Memory planning (allocate PE memory, insert halo exchanges)        │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 5. CSL CODE GENERATION                                                     │
│    For each operation:                                                     │
│    ├── Select appropriate CSL template                                     │
│    ├── Substitute parameters                                               │
│    ├── Generate halo exchange code if needed                               │
│    └── Generate layout.csl for PE fabric                                   │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 6. CSL COMPILATION                                                         │
│    cslc layout.csl --fabric-dims=WxH -o output_dir                        │
│    Output: out.elf (WSE binary)                                            │
└────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 7. EXECUTION                                                               │
│    ├── Transfer input image to WSE (distributed across PEs)               │
│    ├── Execute compiled kernel                                             │
│    ├── Transfer output back to host                                        │
│    └── Store result in output tensor                                       │
└────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Kernel Caching

Compiled kernels are cached for reuse:

```
Cache Key = hash(
    operation_type,
    input_shape,
    output_shape,
    parameters,
    layout
)

if cache[key] exists:
    return cached_kernel
else:
    kernel = generate_and_compile()
    cache[key] = kernel
    return kernel
```

---

## 8. Error Handling

### 8.1 Validation Points

| Stage | Validation |
|-------|------------|
| Transform construction | Parameter ranges, types |
| Transform application | Input is valid image (4D NCHW) |
| Crop operations | Crop size ≤ image size |
| Normalize | Mean/std length matches channels |
| Layout planning | Tile fits in PE memory |
| CSL generation | Generated code is syntactically valid |
| Compilation | cslc returns success |
| Execution | No runtime errors from SDK |

### 8.2 Error Messages

```cpp
// Example error messages
"Invalid image tensor: expected 4D NCHW format, got shape [3, 64, 64]"
"Crop size (256, 256) larger than image size (224, 224)"
"Normalize mean length (4) does not match number of channels (3)"
"PE memory exceeded: tile requires 65536 bytes, only 32768 available"
```

---

## 9. Performance Considerations

### 9.1 Optimization Opportunities

| Optimization | Description | Impact |
|--------------|-------------|--------|
| **Transform Fusion** | Combine resize + normalize into single kernel | Reduce memory bandwidth |
| **Tile Size Tuning** | Adjust tile size for operation characteristics | Maximize PE utilization |
| **Halo Minimization** | Use larger tiles to reduce halo overhead | Less communication |
| **Double Buffering** | Overlap compute and communication | Hide latency |
| **Layout Matching** | Keep tensors in compatible layouts | Avoid redistribution |

### 9.2 Memory Budget

```
Per-PE Memory Budget (48KB total):
├── Code/Text:     8 KB
├── Stack:         4 KB
├── Routing:       2 KB
└── Data:         34 KB available
    ├── Input tile + halo
    ├── Output tile
    └── Scratch buffers
```

---

## 10. Future Extensibility

### 10.1 Completed in Phase 2

| Component | Status | Description |
|-----------|--------|-------------|
| **nn.Module** | ✅ Complete | Base class for all neural network layers |
| **nn.Sequential** | ✅ Complete | Container for chaining modules |
| **nn.Conv2d** | ✅ Complete | 2D convolution with groups, dilation support |
| **nn.BatchNorm2d** | ✅ Complete | Batch normalization for 2D data |
| **nn.ReLU/SiLU** | ✅ Complete | Activation functions |
| **nn.MaxPool2d/AvgPool2d** | ✅ Complete | Pooling layers |
| **nn.Linear** | ✅ Complete | Fully connected layer |
| **ResNet** | ✅ Complete | ResNet-18/34/50/101/152, ResNeXt, WideResNet |
| **EfficientNet** | ✅ Complete | EfficientNet-B0 through B4 |

### 10.2 Completed in Phase 3

| Component | Status | Description |
|-----------|--------|-------------|
| **RandomTransform** | ✅ Complete | Thread-safe base class with secure RNG |
| **RandomHorizontalFlip** | ✅ Complete | Probability-based horizontal flip |
| **RandomVerticalFlip** | ✅ Complete | Probability-based vertical flip |
| **RandomRotation** | ✅ Complete | Angle-based rotation with expand support |
| **ColorJitter** | ✅ Complete | Brightness, contrast, saturation, hue adjustment |
| **GaussianBlur** | ✅ Complete | Separable Gaussian blur with variable kernel |
| **Security Limits** | ✅ Complete | Validation for rotation (360°), blur (31 kernel, 10σ), color (2x factor) |
| **CSL Templates** | ✅ Complete | hflip, vflip, rotate, color_jitter, gaussian_blur |

### 10.3 Completed in Phase 4

| Component | Status | Description |
|-----------|--------|-------------|
| **GridSample** | ✅ Complete | Coordinate-based spatial sampling with bilinear/nearest modes |
| **ROIAlign** | ✅ Complete | Region of Interest feature extraction for detection models |
| **NMS** | ✅ Complete | Non-Maximum Suppression for box filtering |
| **BatchedNMS** | ✅ Complete | Class-aware NMS per batch item |
| **SoftNMS** | ✅ Complete | Score decay with linear/gaussian methods |
| **DetectionBox** | ✅ Complete | Box representation with IoU computation |
| **ROI** | ✅ Complete | Region of Interest specification |
| **Security Limits** | ✅ Complete | Bounds validation for ops (MAX_ROIS, MAX_NMS_BOXES, etc.) |
| **CSL Templates** | ✅ Complete | grid_sample, roi_align, nms templates |

### 10.4 Completed in Phase 5

| Component | Status | Description |
|-----------|--------|-------------|
| **io.read_image** | ✅ Complete | Load image files to [C, H, W] numpy arrays |
| **io.write_image** | ✅ Complete | Save [C, H, W] arrays to image files |
| **io.decode_image** | ✅ Complete | Decode image bytes to tensors |
| **io.encode_image** | ✅ Complete | Encode tensors to image bytes |
| **Dataset** | ✅ Complete | Base class for map-style datasets |
| **IterableDataset** | ✅ Complete | Base class for streaming datasets |
| **VisionDataset** | ✅ Complete | Base class for image datasets with transforms |
| **ImageFolder** | ✅ Complete | Directory-based image loading with class discovery |
| **DataLoader** | ✅ Complete | Batched loading with single/multi-process workers |
| **SequentialSampler** | ✅ Complete | Sequential index iteration |
| **RandomSampler** | ✅ Complete | Random permutation or with replacement |
| **SubsetRandomSampler** | ✅ Complete | Random sampling from subset |
| **WeightedRandomSampler** | ✅ Complete | Probability-weighted sampling |
| **BatchSampler** | ✅ Complete | Wrap sampler to yield batches |
| **default_collate** | ✅ Complete | Auto-collate based on element type |
| **pad_collate** | ✅ Complete | Pad variable-size tensors to max size |
| **Security Limits** | ✅ Complete | Path validation, size limits, traversal prevention |

See [PHASE5_IMPLEMENTATION_GUIDE.md](PHASE5_IMPLEMENTATION_GUIDE.md) for detailed architecture.

### 10.5 Phase 6+ Operations

| Operation | CSL Considerations |
|-----------|-------------------|
| **Dropout** | Stochastic operation, requires RNG |
| **RandomAffine** | Combined rotation, translation, scale, shear |
| **RandomPerspective** | 3x3 projective transformation |
| **RandomErasing** | Region masking for regularization |
| **Mixup/CutMix** | Cross-sample augmentation |

### 10.7 Future Model Architectures

| Model | Vision Components Needed |
|-------|-------------------------|
| **ViT** | Patch embedding, attention (from PyFlame) |
| **MobileNet** | Depthwise separable convolutions |
| **ConvNeXt** | Layer normalization, GELU activation |

### 10.8 Extension Points

- Custom interpolation modes
- Custom color transforms
- Model-specific preprocessing pipelines
- Dataset integrations
- Pretrained weight loading from torchvision
- Custom model architectures

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **CSL** | Cerebras Software Language - programming language for WSE |
| **WSE** | Wafer-Scale Engine - Cerebras's AI accelerator chip |
| **PE** | Processing Element - individual compute unit on WSE |
| **NCHW** | Tensor format: Batch, Channels, Height, Width |
| **Halo** | Border pixels from neighboring PEs needed for computation |
| **Wavelet** | 32-bit message for inter-PE communication |
| **Lazy Evaluation** | Deferring computation until result is needed |
| **IR** | Intermediate Representation - computation graph |

---

*Document Version: 6.0*
*Author: PyFlameVision Team*
*Last Updated: January 12, 2026*

---

## Appendix B: Project Configuration

### Package Structure

```
pyflame-vision/
├── pyproject.toml          # Modern Python packaging configuration
├── setup.py                # C++ extension build with CMake
├── requirements.txt        # Core dependencies
├── requirements-dev.txt    # Development dependencies
├── MANIFEST.in             # Source distribution manifest
├── README.md               # Project README
├── LICENSE                 # MIT License
├── CMakeLists.txt          # Root CMake configuration
├── python/
│   ├── CMakeLists.txt      # Python bindings CMake
│   ├── bindings.cpp        # pybind11 bindings
│   └── pyflame_vision/     # Python package
│       ├── __init__.py
│       ├── _version.py
│       ├── transforms/
│       ├── datasets/
│       ├── io/
│       └── ops/
├── include/pyflame_vision/ # C++ headers
├── src/                    # C++ source
├── tests/
│   ├── python/             # Python tests
│   └── cpp/                # C++ tests
├── examples/
└── docs/
```

### Build Configuration

The CMake build system supports both standalone and integrated modes:

```bash
# Standalone build (no PyFlame)
cmake -B build -DPYFLAME_VISION_STANDALONE=ON
cmake --build build

# Integrated build with PyFlame
cmake -B build -DPYFLAME_DIR=/path/to/pyflame
cmake --build build
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `PYFLAME_DIR` | Path to PyFlame source/installation |
| `PYFLAME_VISION_PURE_PYTHON` | Skip C++ extension build |
| `DEBUG` | Build in debug mode |
