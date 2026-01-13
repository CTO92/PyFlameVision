# CSL Template Development Guide

> **PRE-RELEASE ALPHA SOFTWARE** - This project is currently in an early alpha stage. APIs may change without notice.

This guide covers how to create and modify CSL (Cerebras Software Language) templates for PyFlameVision transforms.

## Overview

CSL templates are parameterized code that generates optimized kernels for the Cerebras Wafer-Scale Engine (WSE). PyFlameVision uses templates to generate operation-specific code at compile time.

### Template Location

```
src/backend/csl_templates/
├── resize_bilinear.csl.template    # Bilinear resize kernel
├── normalize.csl.template          # Normalization kernel
├── crop.csl.template               # Crop kernel
└── halo_exchange.csl.template      # Halo exchange for distributed ops
```

---

## Template Syntax

### Parameter Substitution

Templates use `{{PARAMETER}}` syntax for substitution:

```csl
// Template
const TILE_H: u32 = {{TILE_H}};
const TILE_W: u32 = {{TILE_W}};
const T = {{DTYPE}};

// Generated (example)
const TILE_H: u32 = 64;
const TILE_W: u32 = 64;
const T = f32;
```

### Common Parameters

| Parameter | Description | Example Values |
|-----------|-------------|----------------|
| `{{TILE_H}}` | Tile height | 32, 64, 128 |
| `{{TILE_W}}` | Tile width | 32, 64, 128 |
| `{{NUM_CHANNELS}}` | Number of channels | 1, 3, 4 |
| `{{DTYPE}}` | Data type | f32, f16, i32 |
| `{{PE_ROWS}}` | PE grid rows | 1, 2, 4, 8 |
| `{{PE_COLS}}` | PE grid columns | 1, 2, 4, 8 |
| `{{HALO_SIZE}}` | Halo width for exchange | 1, 2 |
| `{{TIMESTAMP}}` | Generation timestamp | ISO 8601 string |

---

## Template Structure

A typical CSL template has these sections:

### 1. Header and Imports

```csl
// PyFlame Vision Generated - {{TRANSFORM_NAME}}
// Generated: {{TIMESTAMP}}

param memcpy = @import_module("<memcpy/get_params>", .{
    .width = {{PE_COLS}},
    .height = {{PE_ROWS}},
});

const sys_mod = @import_module("<memcpy_multi/memcpy>", memcpy);
```

### 2. Parameters and Constants

```csl
// PE coordinates (set per-PE)
param pe_x: u32;
param pe_y: u32;

// Compile-time constants from template
const TILE_H: u32 = {{TILE_H}};
const TILE_W: u32 = {{TILE_W}};
const NUM_CHANNELS: u32 = {{NUM_CHANNELS}};
const T = {{DTYPE}};
```

### 3. Data Buffers

```csl
// Input data buffer
var input: [NUM_CHANNELS][TILE_H][TILE_W]T =
    @zeros([NUM_CHANNELS][TILE_H][TILE_W]T);

// Output data buffer
var output: [NUM_CHANNELS][TILE_H][TILE_W]T =
    @zeros([NUM_CHANNELS][TILE_H][TILE_W]T);

// Export for host access
export var input_ptr = &input;
export var output_ptr = &output;
```

### 4. Compute Tasks

```csl
task compute() void {
    for (@range(u32, 0, NUM_CHANNELS)) |c| {
        for (@range(u32, 0, TILE_H)) |h| {
            for (@range(u32, 0, TILE_W)) |w| {
                // Compute operation
                output[c][h][w] = process(input[c][h][w]);
            }
        }
    }
    sys_mod.unblock_cmd_stream();
}

comptime {
    @bind_local_task_to_color(compute, sys_mod.LAUNCH);
}
```

---

## Creating a New Template

### Step 1: Define Parameters

Identify what needs to be parameterized:

```cpp
// In C++ transform class
struct GaussianBlurParams {
    int kernel_size;      // -> {{KERNEL_SIZE}}
    float sigma;          // -> {{SIGMA}}
    int tile_height;      // -> {{TILE_H}}
    int tile_width;       // -> {{TILE_W}}
    int num_channels;     // -> {{NUM_CHANNELS}}
    std::string dtype;    // -> {{DTYPE}}
};
```

### Step 2: Create Template File

Create `src/backend/csl_templates/gaussian_blur.csl.template`:

```csl
// PyFlame Vision Generated - Gaussian Blur
// Kernel Size: {{KERNEL_SIZE}}, Sigma: {{SIGMA}}
// Generated: {{TIMESTAMP}}

param memcpy = @import_module("<memcpy/get_params>", .{
    .width = {{PE_COLS}},
    .height = {{PE_ROWS}},
});

const sys_mod = @import_module("<memcpy_multi/memcpy>", memcpy);

param pe_x: u32;
param pe_y: u32;

const TILE_H: u32 = {{TILE_H}};
const TILE_W: u32 = {{TILE_W}};
const NUM_CHANNELS: u32 = {{NUM_CHANNELS}};
const KERNEL_SIZE: u32 = {{KERNEL_SIZE}};
const HALO: u32 = KERNEL_SIZE / 2;
const T = {{DTYPE}};

// Data with halo for convolution
var data: [NUM_CHANNELS][TILE_H + 2*HALO][TILE_W + 2*HALO]T =
    @zeros([NUM_CHANNELS][TILE_H + 2*HALO][TILE_W + 2*HALO]T);

var output: [NUM_CHANNELS][TILE_H][TILE_W]T =
    @zeros([NUM_CHANNELS][TILE_H][TILE_W]T);

// Precomputed Gaussian kernel (1D, will be separated)
var kernel: [KERNEL_SIZE]T = .{ {{KERNEL_WEIGHTS}} };

export var data_ptr = &data;
export var output_ptr = &output;

task gaussian_blur() void {
    // Separable Gaussian blur implementation
    for (@range(u32, 0, NUM_CHANNELS)) |c| {
        // Horizontal pass
        for (@range(u32, 0, TILE_H)) |h| {
            for (@range(u32, 0, TILE_W)) |w| {
                var sum: T = 0.0;
                for (@range(u32, 0, KERNEL_SIZE)) |k| {
                    sum += data[c][h + HALO][w + k] * kernel[k];
                }
                output[c][h][w] = sum;
            }
        }

        // Vertical pass (in-place on output)
        // ... similar loop structure
    }

    sys_mod.unblock_cmd_stream();
}

comptime {
    @bind_local_task_to_color(gaussian_blur, sys_mod.LAUNCH);
}
```

### Step 3: Implement Template Generator

Use the secure `CSLGenerator` class from `include/pyflame_vision/backend/csl_generator.hpp`:

```cpp
#include "pyflame_vision/backend/csl_generator.hpp"

using namespace pyflame_vision::backend;

// Method 1: Using CSLGenerator class
CSLGenerator gen;
gen.set_numeric("width", 256);
gen.set_numeric("height", 256);
gen.set_dtype("dtype", "f32");
gen.set_boolean("aligned", true);
std::string csl = gen.generate_from_file("templates/my_op.csl.template");

// Method 2: Using CSLBuilder fluent API
std::string csl = CSLBuilder()
    .set("width", 256)
    .set("height", 256)
    .set_dtype("dtype", "f32")
    .set_bool("aligned", true)
    .generate_from_file("templates/my_op.csl.template");
```

**IMPORTANT: Security Note**

The `CSLGenerator` validates ALL parameters to prevent code injection:

- **Numeric values**: Only digits, decimal point, sign, and exponent characters allowed
- **Identifiers**: Must start with letter/underscore, only alphanumeric and underscore
- **Dtypes**: Whitelist validated (f32, f16, i32, etc.)
- **Booleans**: Only "true" or "false"

```cpp
// These will throw TemplateError (code injection attempt)
gen.set_param("bad", TemplateParam::numeric("123; @import"));  // FAILS
gen.set_param("bad", TemplateParam::identifier("rm -rf"));     // FAILS

// Missing placeholders also detected
gen.set_numeric("width", 256);
// Forgot to set "height"
gen.generate("const W = ${width}; const H = ${height};");  // FAILS: Unsubstituted
```

### Step 4: Integrate with Transform

```cpp
// src/transforms/gaussian_blur.cpp
#include "pyflame_vision/transforms/gaussian_blur.hpp"
#include "pyflame_vision/backend/csl_generator.hpp"

namespace pyflame_vision::transforms {

std::string GaussianBlur::generate_csl(
    const std::vector<int64_t>& input_shape,
    const pyflame::MeshLayout& layout
) const {
    auto template_content = backend::CSLGenerator::load_template(
        "csl_templates/gaussian_blur.csl.template"
    );

    // Compute kernel weights
    auto weights = compute_gaussian_weights(kernel_size_, sigma_);
    std::string weights_str = format_weights(weights);

    backend::CSLGenerator::Params params = {
        {"TILE_H", std::to_string(compute_tile_height(input_shape, layout))},
        {"TILE_W", std::to_string(compute_tile_width(input_shape, layout))},
        {"NUM_CHANNELS", std::to_string(input_shape[1])},
        {"KERNEL_SIZE", std::to_string(kernel_size_)},
        {"SIGMA", std::to_string(sigma_)},
        {"KERNEL_WEIGHTS", weights_str},
        {"DTYPE", dtype_to_csl(dtype_)},
        {"PE_ROWS", std::to_string(layout.pe_rows)},
        {"PE_COLS", std::to_string(layout.pe_cols)},
        {"TIMESTAMP", current_timestamp()},
    };

    return backend::CSLGenerator::generate(template_content, params);
}

}
```

---

## Distributed Operations

### Halo Exchange Pattern

For operations requiring neighboring pixels (convolutions, pooling):

```csl
// Data with halo regions
var data: [C][H + 2*HALO][W + 2*HALO]T;

// Halo exchange colors
const color_north: color = @get_color(0);
const color_south: color = @get_color(1);
const color_east: color = @get_color(2);
const color_west: color = @get_color(3);

task exchange_halos() void {
    // 1. Pack border data into send buffers
    pack_halos();

    // 2. Send to neighbors
    if (pe_y > 0) send_north();
    if (pe_y < GRID_ROWS - 1) send_south();
    if (pe_x > 0) send_west();
    if (pe_x < GRID_COLS - 1) send_east();

    // 3. Receive from neighbors
    if (pe_y > 0) receive_from_north();
    if (pe_y < GRID_ROWS - 1) receive_from_south();
    if (pe_x > 0) receive_from_west();
    if (pe_x < GRID_COLS - 1) receive_from_east();

    sys_mod.unblock_cmd_stream();
}
```

### Memory Layout

```
Interior + Halo layout for 3x3 kernel (HALO=1):

    +---+---+---+---+---+
    | H | H | H | H | H |  <- Top halo (from north neighbor)
    +---+---+---+---+---+
    | H | I | I | I | H |  <- Left/Right halos
    +---+---+---+---+---+
    | H | I | I | I | H |  <- I = Interior (local data)
    +---+---+---+---+---+
    | H | I | I | I | H |
    +---+---+---+---+---+
    | H | H | H | H | H |  <- Bottom halo (from south neighbor)
    +---+---+---+---+---+
```

---

## Optimization Techniques

### 1. Loop Unrolling

```csl
// For small, fixed-size loops
comptime {
    // Unroll 3x3 convolution
    for (comptime_int(0, 3)) |kh| {
        for (comptime_int(0, 3)) |kw| {
            sum += data[c][h + kh][w + kw] * kernel[kh][kw];
        }
    }
}
```

### 2. Vectorization

```csl
// Use SIMD when possible (depends on data type and alignment)
const VECTOR_WIDTH = 4;  // For f32

for (@range(u32, 0, TILE_W, VECTOR_WIDTH)) |w| {
    // Process 4 elements at once
    @vector_op(data[c][h][w:w+VECTOR_WIDTH]);
}
```

### 3. Memory Access Optimization

```csl
// Coalesce memory accesses
// Bad: strided access
for (h) { for (w) { for (c) { access(c, h, w); }}}

// Good: sequential access
for (c) { for (h) { for (w) { access(c, h, w); }}}
```

### 4. Reduce Memory Footprint

```csl
// Use appropriate data types
const T = f16;  // Instead of f32 when precision allows

// Share buffers when possible
var buffer: [MAX_SIZE]T;  // Reuse for multiple operations
```

---

## Testing Templates

### Unit Test for Generated Code

```cpp
TEST(CSLTemplateTest, ResizeGeneratesValidCSL) {
    Resize resize(224);

    std::vector<int64_t> input_shape = {1, 3, 480, 640};
    auto layout = pyflame::MeshLayout::Grid(2, 2);

    std::string csl = resize.generate_csl(input_shape, layout);

    // Check required elements
    EXPECT_TRUE(csl.find("const TILE_H") != std::string::npos);
    EXPECT_TRUE(csl.find("const TILE_W") != std::string::npos);
    EXPECT_TRUE(csl.find("task ") != std::string::npos);

    // Check no unsubstituted placeholders
    EXPECT_EQ(csl.find("{{"), std::string::npos);
}
```

### Validation Checklist

- [ ] All placeholders are substituted
- [ ] Generated code compiles with CSL compiler
- [ ] Memory usage fits within PE limits
- [ ] Halo exchange is correct for distributed ops
- [ ] Edge cases handled (boundary PEs, last tiles)

---

## Best Practices

1. **Comment generated code** - Include parameters in header comments
2. **Validate parameters** - Check constraints before generation
3. **Handle edge cases** - Boundary conditions, odd dimensions
4. **Minimize memory** - Reuse buffers, use appropriate dtypes
5. **Test thoroughly** - Unit test generation, validate output
6. **Document assumptions** - PE memory limits, alignment requirements

---

## See Also

- [Architecture Overview](../ARCHITECTURE.md)
- [Testing Guide](./TESTING.md)
- [Cerebras CSL Documentation](https://docs.cerebras.net/csl/)
