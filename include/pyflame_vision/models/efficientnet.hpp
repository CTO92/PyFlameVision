#pragma once

/// @file efficientnet.hpp
/// @brief EfficientNet family of models
///
/// Provides EfficientNet-B0 through B4 with optional pretrained weights.
/// Based on "EfficientNet: Rethinking Model Scaling for CNNs" (Tan & Le, 2019)

#include "pyflame_vision/nn/nn.hpp"
#include <cmath>

namespace pyflame_vision::models {

/// Squeeze-and-Excitation block
///
/// Channel attention mechanism that recalibrates channel-wise feature responses.
/// SE(x) = x * sigmoid(fc2(relu(fc1(gap(x)))))
///
/// @note Thread Safety: SqueezeExcitation is thread-safe (immutable after construction).
class SqueezeExcitation : public nn::Module {
public:
    /// Create SE block
    /// @param input_channels Number of input channels
    /// @param squeeze_channels Number of squeezed channels (reduction)
    SqueezeExcitation(int64_t input_channels, int64_t squeeze_channels);

    nn::TensorSpec forward(const nn::TensorSpec& input) const override;
    std::string name() const override { return "SqueezeExcitation"; }
    std::string repr() const override;
    std::vector<nn::Parameter> parameters() const override;
    std::vector<std::shared_ptr<nn::Module>> children() const override;

    int64_t input_channels() const { return input_channels_; }
    int64_t squeeze_channels() const { return squeeze_channels_; }

private:
    int64_t input_channels_;
    int64_t squeeze_channels_;

    std::shared_ptr<nn::AdaptiveAvgPool2d> avgpool_;
    std::shared_ptr<nn::Conv2d> fc1_;
    std::shared_ptr<nn::SiLU> activation_;
    std::shared_ptr<nn::Conv2d> fc2_;
    std::shared_ptr<nn::Sigmoid> scale_;
};

/// MBConv block configuration
struct MBConvConfig {
    int64_t expand_ratio;      ///< Expansion ratio for inverted bottleneck (1, 4, or 6)
    int64_t kernel_size;       ///< Depthwise conv kernel size (3 or 5)
    int64_t stride;            ///< Stride for depthwise conv (1 or 2)
    int64_t input_channels;    ///< Number of input channels
    int64_t out_channels;      ///< Number of output channels
    int64_t num_layers;        ///< Number of times to repeat this block

    /// Get expanded channels
    int64_t expanded_channels() const {
        return input_channels * expand_ratio;
    }
};

/// Mobile Inverted Bottleneck Convolution block
///
/// The core building block of EfficientNet.
/// Architecture: expand 1x1 -> depthwise 3x3/5x5 -> SE -> project 1x1 -> residual
///
/// @note Thread Safety: MBConv is thread-safe (immutable after construction).
class MBConv : public nn::Module {
public:
    /// Create MBConv block
    /// @param config Block configuration
    /// @param stochastic_depth_prob Stochastic depth probability (for training)
    MBConv(const MBConvConfig& config, float stochastic_depth_prob = 0.0f);

    nn::TensorSpec forward(const nn::TensorSpec& input) const override;
    std::string name() const override { return "MBConv"; }
    std::string repr() const override;
    std::vector<nn::Parameter> parameters() const override;
    std::vector<std::shared_ptr<nn::Module>> children() const override;

    /// Check if residual connection is used
    bool use_residual() const { return use_residual_; }

    /// Get configuration
    const MBConvConfig& config() const { return config_; }

private:
    MBConvConfig config_;
    bool use_residual_;
    float stochastic_depth_prob_;

    // Expansion phase (if expand_ratio > 1)
    std::shared_ptr<nn::Conv2d> expand_conv_;
    std::shared_ptr<nn::BatchNorm2d> expand_bn_;
    std::shared_ptr<nn::SiLU> expand_act_;

    // Depthwise convolution
    std::shared_ptr<nn::Conv2d> depthwise_conv_;
    std::shared_ptr<nn::BatchNorm2d> depthwise_bn_;
    std::shared_ptr<nn::SiLU> depthwise_act_;

    // Squeeze-and-Excitation
    std::shared_ptr<SqueezeExcitation> se_;

    // Project phase (linear bottleneck, no activation)
    std::shared_ptr<nn::Conv2d> project_conv_;
    std::shared_ptr<nn::BatchNorm2d> project_bn_;
};

/// EfficientNet model configuration
struct EfficientNetConfig {
    std::string name;           ///< Model name (e.g., "efficientnet_b0")
    float width_mult;           ///< Width multiplier (alpha)
    float depth_mult;           ///< Depth multiplier (beta)
    int64_t input_size;         ///< Expected input size (e.g., 224)
    float dropout;              ///< Dropout probability for classifier

    /// Base MBConv configurations (before scaling)
    /// These are scaled by width_mult and depth_mult
    std::vector<MBConvConfig> block_configs;

    /// Get default block configurations (B0 baseline)
    static std::vector<MBConvConfig> default_block_configs();
};

/// EfficientNet model
///
/// Compound scaling CNN that uniformly scales depth, width, and resolution.
///
/// Input: [N, 3, H, W] - expects normalized RGB images
/// Output: [N, num_classes]
///
/// @note Thread Safety: EfficientNet is thread-safe (immutable after construction).
class EfficientNet : public nn::Module {
public:
    /// Create EfficientNet with specified configuration
    /// @param config Model configuration (use factory functions)
    /// @param num_classes Number of output classes (0 to remove classifier)
    explicit EfficientNet(const EfficientNetConfig& config, int64_t num_classes = 1000);

    nn::TensorSpec forward(const nn::TensorSpec& input) const override;
    std::string name() const override { return "EfficientNet"; }
    std::string repr() const override;
    std::vector<nn::Parameter> parameters() const override;
    std::map<std::string, nn::Parameter> named_parameters(
        const std::string& prefix = ""
    ) const override;
    std::vector<std::shared_ptr<nn::Module>> children() const override;

    /// Get feature map (before classifier)
    nn::TensorSpec forward_features(const nn::TensorSpec& input) const;

    /// Remove classifier for feature extraction
    void remove_classifier() { classifier_ = nullptr; }

    /// Check if model has classifier
    bool has_classifier() const { return classifier_ != nullptr; }

    /// Get configuration
    const EfficientNetConfig& config() const { return config_; }

    /// Get number of features before classifier
    int64_t num_features() const { return last_channels_; }

private:
    EfficientNetConfig config_;
    int64_t num_classes_;
    int64_t last_channels_;

    // Stem
    std::shared_ptr<nn::Conv2d> stem_conv_;
    std::shared_ptr<nn::BatchNorm2d> stem_bn_;
    std::shared_ptr<nn::SiLU> stem_act_;

    // MBConv blocks (features)
    std::shared_ptr<nn::Sequential> features_;

    // Head
    std::shared_ptr<nn::Conv2d> head_conv_;
    std::shared_ptr<nn::BatchNorm2d> head_bn_;
    std::shared_ptr<nn::SiLU> head_act_;
    std::shared_ptr<nn::AdaptiveAvgPool2d> avgpool_;
    std::shared_ptr<nn::Linear> classifier_;

    /// Adjust channels based on width multiplier
    static int64_t adjust_channels(int64_t channels, float width_mult);

    /// Adjust depth (number of layers) based on depth multiplier
    static int64_t adjust_depth(int64_t num_layers, float depth_mult);

    void build_features();
    void validate_params() const;
};

// ============================================================================
// Configuration factory functions
// ============================================================================

/// Get EfficientNet-B0 configuration (baseline)
EfficientNetConfig efficientnet_b0_config();

/// Get EfficientNet-B1 configuration
EfficientNetConfig efficientnet_b1_config();

/// Get EfficientNet-B2 configuration
EfficientNetConfig efficientnet_b2_config();

/// Get EfficientNet-B3 configuration
EfficientNetConfig efficientnet_b3_config();

/// Get EfficientNet-B4 configuration
EfficientNetConfig efficientnet_b4_config();

// ============================================================================
// Model factory functions (matching torchvision API)
// ============================================================================

/// Create EfficientNet-B0
/// @param num_classes Number of output classes (default 1000 for ImageNet)
/// @param pretrained Load pretrained weights (not implemented yet)
std::shared_ptr<EfficientNet> efficientnet_b0(int64_t num_classes = 1000, bool pretrained = false);

/// Create EfficientNet-B1
std::shared_ptr<EfficientNet> efficientnet_b1(int64_t num_classes = 1000, bool pretrained = false);

/// Create EfficientNet-B2
std::shared_ptr<EfficientNet> efficientnet_b2(int64_t num_classes = 1000, bool pretrained = false);

/// Create EfficientNet-B3
std::shared_ptr<EfficientNet> efficientnet_b3(int64_t num_classes = 1000, bool pretrained = false);

/// Create EfficientNet-B4
std::shared_ptr<EfficientNet> efficientnet_b4(int64_t num_classes = 1000, bool pretrained = false);

}  // namespace pyflame_vision::models
