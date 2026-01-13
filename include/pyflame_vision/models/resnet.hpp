#pragma once

/// @file resnet.hpp
/// @brief ResNet family of models
///
/// Provides ResNet-18, ResNet-34, ResNet-50, ResNet-101, ResNet-152
/// with optional pretrained weights.

#include "pyflame_vision/nn/nn.hpp"
#include <optional>

namespace pyflame_vision::models {

/// ResNet block type
enum class ResNetBlockType : uint8_t {
    BASIC,      ///< For ResNet-18, ResNet-34 (2 convolutions)
    BOTTLENECK  ///< For ResNet-50, ResNet-101, ResNet-152 (3 convolutions)
};

/// Basic residual block (2 convolutions)
///
/// Used in ResNet-18 and ResNet-34.
/// Architecture: conv3x3 -> bn -> relu -> conv3x3 -> bn -> (+residual) -> relu
///
/// @note Thread Safety: BasicBlock is thread-safe (immutable after construction).
class BasicBlock : public nn::Module {
public:
    static constexpr int expansion = 1;

    /// Create BasicBlock
    /// @param inplanes Number of input channels
    /// @param planes Number of output channels (before expansion)
    /// @param stride Stride for first convolution
    /// @param downsample Optional downsample module for residual
    /// @param groups Number of groups (for ResNeXt)
    /// @param base_width Base width per group
    /// @param dilation Dilation for convolutions
    BasicBlock(
        int64_t inplanes,
        int64_t planes,
        int64_t stride = 1,
        std::shared_ptr<nn::Module> downsample = nullptr,
        int64_t groups = 1,
        int64_t base_width = 64,
        int64_t dilation = 1
    );

    nn::TensorSpec forward(const nn::TensorSpec& input) const override;
    std::string name() const override { return "BasicBlock"; }
    std::string repr() const override;
    std::vector<nn::Parameter> parameters() const override;
    std::vector<std::shared_ptr<nn::Module>> children() const override;

    // Accessors
    int64_t inplanes() const { return inplanes_; }
    int64_t planes() const { return planes_; }
    int64_t stride() const { return stride_; }

private:
    int64_t inplanes_;
    int64_t planes_;
    int64_t stride_;

    std::shared_ptr<nn::Conv2d> conv1_;
    std::shared_ptr<nn::BatchNorm2d> bn1_;
    std::shared_ptr<nn::ReLU> relu_;
    std::shared_ptr<nn::Conv2d> conv2_;
    std::shared_ptr<nn::BatchNorm2d> bn2_;
    std::shared_ptr<nn::Module> downsample_;
};

/// Bottleneck residual block (3 convolutions with 1x1 reduction)
///
/// Used in ResNet-50, ResNet-101, ResNet-152.
/// Architecture: conv1x1 -> bn -> relu -> conv3x3 -> bn -> relu -> conv1x1 -> bn -> (+residual) -> relu
///
/// @note Thread Safety: Bottleneck is thread-safe (immutable after construction).
class Bottleneck : public nn::Module {
public:
    static constexpr int expansion = 4;

    /// Create Bottleneck
    /// @param inplanes Number of input channels
    /// @param planes Number of output channels (before expansion)
    /// @param stride Stride for 3x3 convolution
    /// @param downsample Optional downsample module for residual
    /// @param groups Number of groups (for ResNeXt)
    /// @param base_width Base width per group
    /// @param dilation Dilation for 3x3 convolution
    Bottleneck(
        int64_t inplanes,
        int64_t planes,
        int64_t stride = 1,
        std::shared_ptr<nn::Module> downsample = nullptr,
        int64_t groups = 1,
        int64_t base_width = 64,
        int64_t dilation = 1
    );

    nn::TensorSpec forward(const nn::TensorSpec& input) const override;
    std::string name() const override { return "Bottleneck"; }
    std::string repr() const override;
    std::vector<nn::Parameter> parameters() const override;
    std::vector<std::shared_ptr<nn::Module>> children() const override;

    // Accessors
    int64_t inplanes() const { return inplanes_; }
    int64_t planes() const { return planes_; }
    int64_t stride() const { return stride_; }

private:
    int64_t inplanes_;
    int64_t planes_;
    int64_t stride_;
    int64_t width_;

    std::shared_ptr<nn::Conv2d> conv1_;  // 1x1 reduce
    std::shared_ptr<nn::BatchNorm2d> bn1_;
    std::shared_ptr<nn::Conv2d> conv2_;  // 3x3
    std::shared_ptr<nn::BatchNorm2d> bn2_;
    std::shared_ptr<nn::Conv2d> conv3_;  // 1x1 expand
    std::shared_ptr<nn::BatchNorm2d> bn3_;
    std::shared_ptr<nn::ReLU> relu_;
    std::shared_ptr<nn::Module> downsample_;
};

/// ResNet model architecture
///
/// Implements the ResNet family (ResNet-18 through ResNet-152) following
/// the torchvision API.
///
/// Input: [N, 3, H, W] - expects normalized RGB images
/// Output: [N, num_classes] (or [N, C, H/32, W/32] without FC)
///
/// @note Thread Safety: ResNet is thread-safe (immutable after construction).
class ResNet : public nn::Module {
public:
    /// Create ResNet with specified configuration
    /// @param block_type Basic or Bottleneck block
    /// @param layers Number of blocks in each of 4 stages [layer1, layer2, layer3, layer4]
    /// @param num_classes Number of output classes (0 to remove FC for feature extraction)
    /// @param zero_init_residual Zero-initialize residual connections
    /// @param groups Groups for 3x3 conv (>1 for ResNeXt)
    /// @param width_per_group Base width per group (64 for standard, 4 for ResNeXt)
    /// @param replace_stride_with_dilation Replace stride with dilation in layers
    ResNet(
        ResNetBlockType block_type,
        std::vector<int64_t> layers,
        int64_t num_classes = 1000,
        bool zero_init_residual = false,
        int64_t groups = 1,
        int64_t width_per_group = 64,
        std::optional<std::vector<bool>> replace_stride_with_dilation = std::nullopt
    );

    nn::TensorSpec forward(const nn::TensorSpec& input) const override;
    std::string name() const override { return "ResNet"; }
    std::string repr() const override;
    std::vector<nn::Parameter> parameters() const override;
    std::map<std::string, nn::Parameter> named_parameters(
        const std::string& prefix = ""
    ) const override;
    std::vector<std::shared_ptr<nn::Module>> children() const override;

    /// Get feature maps from each stage (for feature extraction)
    /// Returns shapes after [conv1, layer1, layer2, layer3, layer4]
    std::vector<nn::TensorSpec> forward_features(const nn::TensorSpec& input) const;

    /// Access individual components
    std::shared_ptr<nn::Conv2d> conv1() const { return conv1_; }
    std::shared_ptr<nn::BatchNorm2d> bn1() const { return bn1_; }
    std::shared_ptr<nn::MaxPool2d> maxpool() const { return maxpool_; }
    const nn::Sequential& layer1() const { return *layer1_; }
    const nn::Sequential& layer2() const { return *layer2_; }
    const nn::Sequential& layer3() const { return *layer3_; }
    const nn::Sequential& layer4() const { return *layer4_; }
    std::shared_ptr<nn::AdaptiveAvgPool2d> avgpool() const { return avgpool_; }
    std::shared_ptr<nn::Linear> fc() const { return fc_; }

    /// Remove classification head for feature extraction
    void remove_fc() { fc_ = nullptr; }

    /// Check if model has classification head
    bool has_fc() const { return fc_ != nullptr; }

    /// Get number of output features before FC
    int64_t num_features() const;

    /// Get the block type
    ResNetBlockType block_type() const { return block_type_; }

    /// Get layer configuration
    const std::vector<int64_t>& layer_config() const { return layers_; }

private:
    ResNetBlockType block_type_;
    std::vector<int64_t> layers_;
    int64_t inplanes_ = 64;
    int64_t dilation_ = 1;
    int64_t groups_;
    int64_t base_width_;
    int64_t num_classes_;

    std::shared_ptr<nn::Conv2d> conv1_;
    std::shared_ptr<nn::BatchNorm2d> bn1_;
    std::shared_ptr<nn::ReLU> relu_;
    std::shared_ptr<nn::MaxPool2d> maxpool_;
    std::shared_ptr<nn::Sequential> layer1_;
    std::shared_ptr<nn::Sequential> layer2_;
    std::shared_ptr<nn::Sequential> layer3_;
    std::shared_ptr<nn::Sequential> layer4_;
    std::shared_ptr<nn::AdaptiveAvgPool2d> avgpool_;
    std::shared_ptr<nn::Linear> fc_;

    /// Make a layer (stage) of the network
    std::shared_ptr<nn::Sequential> make_layer(
        int64_t planes,
        int64_t blocks,
        int64_t stride = 1,
        bool dilate = false
    );

    /// Create a downsample module
    std::shared_ptr<nn::Sequential> make_downsample(
        int64_t inplanes,
        int64_t outplanes,
        int64_t stride
    );

    /// Get expansion factor for current block type
    int expansion() const {
        return block_type_ == ResNetBlockType::BASIC ? 1 : 4;
    }

    void validate_params() const;
};

// ============================================================================
// Factory functions (matching torchvision API)
// ============================================================================

/// Create ResNet-18
/// @param num_classes Number of output classes (default 1000 for ImageNet)
/// @param pretrained Load pretrained weights (not implemented yet)
std::shared_ptr<ResNet> resnet18(int64_t num_classes = 1000, bool pretrained = false);

/// Create ResNet-34
std::shared_ptr<ResNet> resnet34(int64_t num_classes = 1000, bool pretrained = false);

/// Create ResNet-50
std::shared_ptr<ResNet> resnet50(int64_t num_classes = 1000, bool pretrained = false);

/// Create ResNet-101
std::shared_ptr<ResNet> resnet101(int64_t num_classes = 1000, bool pretrained = false);

/// Create ResNet-152
std::shared_ptr<ResNet> resnet152(int64_t num_classes = 1000, bool pretrained = false);

// ============================================================================
// ResNeXt variants (grouped convolutions)
// ============================================================================

/// Create ResNeXt-50 (32x4d)
std::shared_ptr<ResNet> resnext50_32x4d(int64_t num_classes = 1000, bool pretrained = false);

/// Create ResNeXt-101 (32x8d)
std::shared_ptr<ResNet> resnext101_32x8d(int64_t num_classes = 1000, bool pretrained = false);

// ============================================================================
// Wide ResNet variants
// ============================================================================

/// Create Wide ResNet-50-2
std::shared_ptr<ResNet> wide_resnet50_2(int64_t num_classes = 1000, bool pretrained = false);

/// Create Wide ResNet-101-2
std::shared_ptr<ResNet> wide_resnet101_2(int64_t num_classes = 1000, bool pretrained = false);

}  // namespace pyflame_vision::models
