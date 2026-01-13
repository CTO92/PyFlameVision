# PyFlameVision Documentation

Welcome to the PyFlameVision documentation. PyFlameVision is a Cerebras-native computer vision library that provides high-performance image processing transforms and model architectures optimized for the Wafer-Scale Engine (WSE).

## Documentation Overview

### Getting Started

- [Installation Guide](./INSTALLATION.md) - How to build and install PyFlameVision
- [Quick Start](./QUICKSTART.md) - Get up and running in 5 minutes
- [Migration from torchvision](./MIGRATION.md) - Transition your existing code

### Guides

- [Architecture Overview](./ARCHITECTURE.md) - Understanding the system design
- [CSL Template Development](./guides/CSL_TEMPLATES.md) - Creating custom CSL kernels
- [Testing Guide](./guides/TESTING.md) - Running and writing tests
- [Contributing](./CONTRIBUTING.md) - How to contribute to PyFlameVision

### API Reference

- [Core Module](./api/CORE.md) - ImageTensor, ColorSpace, InterpolationMode
- [Transforms](./api/TRANSFORMS.md) - Resize, Crop, Normalize, Compose, and Data Augmentation
- [Functional API](./api/FUNCTIONAL.md) - Stateless transform functions
- [Neural Network (nn) Module](./api/NN.md) - Conv2d, BatchNorm2d, ReLU, Linear, etc.
- [Models](./api/MODELS.md) - ResNet, EfficientNet, and model factory functions
- [Operations (ops) Module](./api/OPS.md) - GridSample, ROIAlign, NMS for detection models
- [Datasets Module](./api/DATASETS.md) - Dataset, DataLoader, ImageFolder, Samplers
- [IO Module](./api/IO.md) - read_image, write_image, decode_image, encode_image

### Design Documents

- [Phase 1 Implementation Guide](./PHASE1_IMPLEMENTATION_GUIDE.md) - Core transforms implementation
- [Phase 2 Architecture](./PHASE2_ARCHITECTURE.md) - Model architectures design
- [Phase 2 Implementation Guide](./PHASE2_IMPLEMENTATION_GUIDE.md) - nn module and models implementation
- [Phase 3 Implementation Guide](./PHASE3_IMPLEMENTATION_GUIDE.md) - Data augmentation transforms
- [Phase 4 Implementation Guide](./PHASE4_IMPLEMENTATION_GUIDE.md) - Specialized operations (GridSample, ROIAlign, NMS)
- [Phase 5 Implementation Guide](./PHASE5_IMPLEMENTATION_GUIDE.md) - Dataset integration (ImageFolder, DataLoader)
- [Security Audit](./SECURITY_AUDIT.md) - Security review and mitigations
- [Phase 4 Security Audit](./PHASE4_SECURITY_AUDIT.md) - Phase 4 security review

## Version

Current version: 5.0.0-alpha

## What's New in 5.0

### Phase 5: Dataset Integration

- **Image I/O**: Complete image loading and saving
  - `read_image(path, mode)` - Load image files to numpy arrays [C, H, W]
  - `write_image(tensor, path, format, quality)` - Save arrays to image files
  - `decode_image(data, mode)` - Decode image bytes in memory
  - `encode_image(tensor, format, quality)` - Encode arrays to bytes
  - `ImageReadMode` - UNCHANGED, GRAY, RGB, RGB_ALPHA modes

- **Dataset Base Classes**: PyTorch-compatible abstractions
  - `Dataset` - Map-style datasets with random access
  - `IterableDataset` - Stream-style datasets with sequential access
  - `VisionDataset` - Base class for image datasets with transforms
  - `ConcatDataset`, `Subset`, `random_split()` - Dataset utilities

- **ImageFolder**: Directory-based dataset loading
  - Automatic class discovery from subdirectory names
  - Configurable image extensions and loaders
  - Security validation for paths and sizes

- **DataLoader**: Batched loading with multiprocessing
  - Single-process and multi-worker iteration
  - Customizable samplers and collate functions
  - `drop_last`, `shuffle`, `prefetch_factor` options

- **Samplers**: Flexible sampling strategies
  - `SequentialSampler` - Sequential order
  - `RandomSampler` - Random permutation or with replacement
  - `SubsetRandomSampler` - Random from subset
  - `WeightedRandomSampler` - Probability-weighted sampling
  - `BatchSampler` - Wrap sampler to yield batches

- **Collate Functions**: Batch creation utilities
  - `default_collate` - Auto-collate based on type
  - `pad_collate` - Pad variable-size tensors
  - `stack_collate`, `no_collate`

## What's New in 4.0

### Phase 4: Specialized Operations

- **Grid Sample**: Coordinate-based spatial sampling
  - `GridSample(mode, padding_mode, align_corners)` - Sample from normalized grid coordinates
  - Supports bilinear and nearest interpolation
  - Padding modes: zeros, border, reflection
  - Equivalent to `torch.nn.functional.grid_sample`

- **ROI Align**: Region of Interest feature extraction
  - `ROIAlign(output_size, spatial_scale, sampling_ratio, aligned)` - Extract fixed-size features from ROIs
  - Sub-pixel accurate sampling for detection models
  - Compatible with Faster R-CNN, Mask R-CNN architectures
  - Equivalent to `torchvision.ops.roi_align`

- **Non-Maximum Suppression (NMS)**: Detection box filtering
  - `NMS(iou_threshold)` - Standard greedy NMS
  - `BatchedNMS(iou_threshold)` - Class-aware per-batch NMS
  - `SoftNMS(sigma, iou_threshold, score_threshold, method)` - Soft suppression with score decay
  - `DetectionBox` - Box representation with IoU computation
  - Equivalent to `torchvision.ops.nms` and `torchvision.ops.batched_nms`

- **Security Limits**: Validated operation bounds
  - MAX_ROIS = 10,000; MAX_ROI_OUTPUT_SIZE = 256
  - MAX_GRID_SAMPLE_SIZE = 4,096; MAX_GRID_COORDINATE = 1e6
  - MAX_NMS_BOXES = 100,000

## What's New in 3.0

### Phase 3: Data Augmentation Transforms

- **Random Transform Infrastructure**: Thread-safe base class with secure RNG
  - `RandomTransform` base class with mutex-protected random number generation
  - Secure seed generation using `generate_secure_seed()`
  - Explicit seeding via `set_seed()` for reproducibility

- **Flip Transforms**: Probability-based image flipping
  - `RandomHorizontalFlip(p=0.5)` - Horizontal flip with configurable probability
  - `RandomVerticalFlip(p=0.5)` - Vertical flip with configurable probability

- **Rotation Transform**: Random rotation within angle range
  - `RandomRotation(degrees)` - Symmetric range [-d, d]
  - `RandomRotation(degrees=(min, max))` - Asymmetric range
  - Support for expand mode, custom interpolation, and fill values

- **Color Augmentation**: Color space transforms
  - `ColorJitter(brightness, contrast, saturation, hue)` - Random color adjustments
  - Random application order for increased augmentation diversity
  - Security limits: max 2x factor for brightness/contrast/saturation, 0.5 for hue

- **Blur Transform**: Gaussian blur for regularization
  - `GaussianBlur(kernel_size, sigma)` - Variable kernel and sigma
  - Separable convolution for efficient O(n) complexity
  - Security limits: max kernel size 31 (must be odd), max sigma 10.0

### Phase 2: Model Architectures

- **Neural Network Module (`nn`)**: PyTorch-compatible layer abstractions
  - `Module`, `Sequential`, `ModuleList`
  - `Conv2d`, `BatchNorm2d`, `LayerNorm`
  - `ReLU`, `SiLU`, `GELU`, `Sigmoid`
  - `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`
  - `Linear`, `Flatten`, `Identity`

- **Model Architectures (`models`)**: Production-ready vision models
  - **ResNet Family**: `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
  - **ResNeXt**: `resnext50_32x4d`, `resnext101_32x8d`
  - **Wide ResNet**: `wide_resnet50_2`, `wide_resnet101_2`
  - **EfficientNet Family**: `efficientnet_b0` through `efficientnet_b4`

## License

PyFlameVision is released under the Apache 2.0 License.
