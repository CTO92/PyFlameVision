/// @file efficientnet.cpp
/// @brief EfficientNet model implementation

#include "pyflame_vision/models/efficientnet.hpp"
#include <sstream>
#include <cmath>

namespace pyflame_vision::models {

// ============================================================================
// SqueezeExcitation Implementation
// ============================================================================

SqueezeExcitation::SqueezeExcitation(int64_t input_channels, int64_t squeeze_channels)
    : input_channels_(input_channels)
    , squeeze_channels_(squeeze_channels)
{
    if (input_channels <= 0) {
        throw core::ValidationError("SqueezeExcitation: input_channels must be positive");
    }
    if (squeeze_channels <= 0) {
        throw core::ValidationError("SqueezeExcitation: squeeze_channels must be positive");
    }

    avgpool_ = std::make_shared<nn::AdaptiveAvgPool2d>(1);
    fc1_ = std::make_shared<nn::Conv2d>(input_channels, squeeze_channels, 1);
    activation_ = std::make_shared<nn::SiLU>(true);
    fc2_ = std::make_shared<nn::Conv2d>(squeeze_channels, input_channels, 1);
    scale_ = std::make_shared<nn::Sigmoid>();
}

nn::TensorSpec SqueezeExcitation::forward(const nn::TensorSpec& input) const {
    // Global average pooling
    auto x = avgpool_->forward(input);  // [N, C, 1, 1]

    // FC layers
    x = fc1_->forward(x);
    x = activation_->forward(x);
    x = fc2_->forward(x);
    x = scale_->forward(x);

    // Output has same shape as input (multiply by scale, same shape preserved)
    return input;  // Shape unchanged after SE attention
}

std::string SqueezeExcitation::repr() const {
    std::ostringstream ss;
    ss << "SqueezeExcitation(\n";
    ss << "  (avgpool): " << avgpool_->repr() << "\n";
    ss << "  (fc1): " << fc1_->repr() << "\n";
    ss << "  (activation): " << activation_->repr() << "\n";
    ss << "  (fc2): " << fc2_->repr() << "\n";
    ss << "  (scale): " << scale_->repr() << "\n";
    ss << ")";
    return ss.str();
}

std::vector<nn::Parameter> SqueezeExcitation::parameters() const {
    std::vector<nn::Parameter> params;

    for (const auto& p : fc1_->parameters()) {
        params.push_back({"fc1." + p.name, p.spec, p.requires_grad});
    }
    for (const auto& p : fc2_->parameters()) {
        params.push_back({"fc2." + p.name, p.spec, p.requires_grad});
    }

    return params;
}

std::vector<std::shared_ptr<nn::Module>> SqueezeExcitation::children() const {
    return {avgpool_, fc1_, activation_, fc2_, scale_};
}

// ============================================================================
// MBConv Implementation
// ============================================================================

MBConv::MBConv(const MBConvConfig& config, float stochastic_depth_prob)
    : config_(config)
    , stochastic_depth_prob_(stochastic_depth_prob)
{
    // Residual connection when stride=1 and input/output channels match
    use_residual_ = (config.stride == 1) && (config.input_channels == config.out_channels);

    int64_t expanded = config.expanded_channels();

    // Expansion phase (only if expand_ratio > 1)
    if (config.expand_ratio > 1) {
        expand_conv_ = std::make_shared<nn::Conv2d>(
            config.input_channels, expanded, 1, 1, 0, 1, 1, false
        );
        expand_bn_ = std::make_shared<nn::BatchNorm2d>(expanded);
        expand_act_ = std::make_shared<nn::SiLU>(true);
    }

    // Depthwise convolution
    int64_t padding = (config.kernel_size - 1) / 2;
    depthwise_conv_ = std::make_shared<nn::Conv2d>(
        expanded, expanded, config.kernel_size, config.stride, padding, 1,
        expanded,  // groups = expanded for depthwise
        false
    );
    depthwise_bn_ = std::make_shared<nn::BatchNorm2d>(expanded);
    depthwise_act_ = std::make_shared<nn::SiLU>(true);

    // Squeeze-and-Excitation
    int64_t squeeze_channels = std::max(int64_t(1), config.input_channels / 4);
    se_ = std::make_shared<SqueezeExcitation>(expanded, squeeze_channels);

    // Project phase (linear bottleneck - no activation)
    project_conv_ = std::make_shared<nn::Conv2d>(
        expanded, config.out_channels, 1, 1, 0, 1, 1, false
    );
    project_bn_ = std::make_shared<nn::BatchNorm2d>(config.out_channels);
}

nn::TensorSpec MBConv::forward(const nn::TensorSpec& input) const {
    auto x = input;

    // Expansion
    if (expand_conv_) {
        x = expand_conv_->forward(x);
        x = expand_bn_->forward(x);
        x = expand_act_->forward(x);
    }

    // Depthwise
    x = depthwise_conv_->forward(x);
    x = depthwise_bn_->forward(x);
    x = depthwise_act_->forward(x);

    // SE
    x = se_->forward(x);

    // Project
    x = project_conv_->forward(x);
    x = project_bn_->forward(x);

    // Residual (if applicable, shape matches)
    // For shape inference, output is from project_bn
    return x;
}

std::string MBConv::repr() const {
    std::ostringstream ss;
    ss << "MBConv(\n";
    if (expand_conv_) {
        ss << "  (expand_conv): " << expand_conv_->repr() << "\n";
        ss << "  (expand_bn): " << expand_bn_->repr() << "\n";
    }
    ss << "  (depthwise_conv): " << depthwise_conv_->repr() << "\n";
    ss << "  (depthwise_bn): " << depthwise_bn_->repr() << "\n";
    ss << "  (se): " << se_->repr() << "\n";
    ss << "  (project_conv): " << project_conv_->repr() << "\n";
    ss << "  (project_bn): " << project_bn_->repr() << "\n";
    ss << "  use_residual=" << (use_residual_ ? "True" : "False") << "\n";
    ss << ")";
    return ss.str();
}

std::vector<nn::Parameter> MBConv::parameters() const {
    std::vector<nn::Parameter> params;

    auto add_params = [&params](const std::shared_ptr<nn::Module>& m, const std::string& prefix) {
        if (m) {
            for (const auto& p : m->parameters()) {
                params.push_back({prefix + "." + p.name, p.spec, p.requires_grad});
            }
        }
    };

    if (expand_conv_) {
        add_params(expand_conv_, "expand_conv");
        add_params(expand_bn_, "expand_bn");
    }
    add_params(depthwise_conv_, "depthwise_conv");
    add_params(depthwise_bn_, "depthwise_bn");
    for (const auto& p : se_->parameters()) {
        params.push_back({"se." + p.name, p.spec, p.requires_grad});
    }
    add_params(project_conv_, "project_conv");
    add_params(project_bn_, "project_bn");

    return params;
}

std::vector<std::shared_ptr<nn::Module>> MBConv::children() const {
    std::vector<std::shared_ptr<nn::Module>> kids;
    if (expand_conv_) {
        kids.push_back(expand_conv_);
        kids.push_back(expand_bn_);
        kids.push_back(expand_act_);
    }
    kids.push_back(depthwise_conv_);
    kids.push_back(depthwise_bn_);
    kids.push_back(depthwise_act_);
    kids.push_back(se_);
    kids.push_back(project_conv_);
    kids.push_back(project_bn_);
    return kids;
}

// ============================================================================
// EfficientNetConfig Implementation
// ============================================================================

std::vector<MBConvConfig> EfficientNetConfig::default_block_configs() {
    // EfficientNet-B0 baseline configuration
    // expand_ratio, kernel, stride, in_ch, out_ch, num_layers
    return {
        {1, 3, 1, 32, 16, 1},     // Stage 1
        {6, 3, 2, 16, 24, 2},     // Stage 2
        {6, 5, 2, 24, 40, 2},     // Stage 3
        {6, 3, 2, 40, 80, 3},     // Stage 4
        {6, 5, 1, 80, 112, 3},    // Stage 5
        {6, 5, 2, 112, 192, 4},   // Stage 6
        {6, 3, 1, 192, 320, 1},   // Stage 7
    };
}

// ============================================================================
// EfficientNet Implementation
// ============================================================================

int64_t EfficientNet::adjust_channels(int64_t channels, float width_mult) {
    // Round to nearest multiple of 8
    int64_t new_channels = static_cast<int64_t>(channels * width_mult);
    int64_t divisor = 8;
    int64_t new_channels_rounded = std::max(
        divisor,
        (new_channels + divisor / 2) / divisor * divisor
    );
    // Make sure rounding doesn't go down by more than 10%
    if (new_channels_rounded < 0.9 * new_channels) {
        new_channels_rounded += divisor;
    }
    return new_channels_rounded;
}

int64_t EfficientNet::adjust_depth(int64_t num_layers, float depth_mult) {
    return static_cast<int64_t>(std::ceil(num_layers * depth_mult));
}

EfficientNet::EfficientNet(const EfficientNetConfig& config, int64_t num_classes)
    : config_(config)
    , num_classes_(num_classes)
{
    validate_params();

    // Stem: 3x3 conv stride 2
    int64_t stem_channels = adjust_channels(32, config_.width_mult);
    stem_conv_ = std::make_shared<nn::Conv2d>(3, stem_channels, 3, 2, 1, 1, 1, false);
    stem_bn_ = std::make_shared<nn::BatchNorm2d>(stem_channels);
    stem_act_ = std::make_shared<nn::SiLU>(true);

    // Build MBConv features
    build_features();

    // Head: 1x1 conv to expand channels
    int64_t last_block_out = adjust_channels(
        config_.block_configs.back().out_channels,
        config_.width_mult
    );
    last_channels_ = adjust_channels(1280, config_.width_mult);

    head_conv_ = std::make_shared<nn::Conv2d>(last_block_out, last_channels_, 1, 1, 0, 1, 1, false);
    head_bn_ = std::make_shared<nn::BatchNorm2d>(last_channels_);
    head_act_ = std::make_shared<nn::SiLU>(true);
    avgpool_ = std::make_shared<nn::AdaptiveAvgPool2d>(1);

    if (num_classes_ > 0) {
        classifier_ = std::make_shared<nn::Linear>(last_channels_, num_classes_);
    }
}

void EfficientNet::validate_params() const {
    if (config_.width_mult <= 0) {
        throw core::ValidationError("EfficientNet: width_mult must be positive");
    }
    if (config_.depth_mult <= 0) {
        throw core::ValidationError("EfficientNet: depth_mult must be positive");
    }
    if (config_.block_configs.empty()) {
        throw core::ValidationError("EfficientNet: block_configs cannot be empty");
    }
    if (num_classes_ < 0) {
        throw core::ValidationError("EfficientNet: num_classes must be non-negative");
    }
}

void EfficientNet::build_features() {
    features_ = std::make_shared<nn::Sequential>();

    // Calculate total blocks for stochastic depth
    int64_t total_blocks = 0;
    for (const auto& cfg : config_.block_configs) {
        total_blocks += adjust_depth(cfg.num_layers, config_.depth_mult);
    }

    int64_t block_idx = 0;
    int64_t in_channels = adjust_channels(32, config_.width_mult);

    for (const auto& base_cfg : config_.block_configs) {
        int64_t out_channels = adjust_channels(base_cfg.out_channels, config_.width_mult);
        int64_t num_layers = adjust_depth(base_cfg.num_layers, config_.depth_mult);

        for (int64_t i = 0; i < num_layers; ++i) {
            MBConvConfig cfg;
            cfg.expand_ratio = base_cfg.expand_ratio;
            cfg.kernel_size = base_cfg.kernel_size;
            // Only first block in stage has stride > 1
            cfg.stride = (i == 0) ? base_cfg.stride : 1;
            cfg.input_channels = (i == 0) ? in_channels : out_channels;
            cfg.out_channels = out_channels;
            cfg.num_layers = 1;

            // Stochastic depth probability increases linearly
            float sd_prob = 0.2f * static_cast<float>(block_idx) / static_cast<float>(total_blocks);

            features_->add(std::make_shared<MBConv>(cfg, sd_prob));
            block_idx++;
        }

        in_channels = out_channels;
    }
}

nn::TensorSpec EfficientNet::forward(const nn::TensorSpec& input) const {
    auto x = forward_features(input);

    if (classifier_) {
        // Flatten: [N, C, 1, 1] -> [N, C]
        x = {{x.shape[0], x.shape[1]}, x.dtype};
        x = classifier_->forward(x);
    }

    return x;
}

nn::TensorSpec EfficientNet::forward_features(const nn::TensorSpec& input) const {
    validate_input_dims(input, 4);

    // Stem
    auto x = stem_conv_->forward(input);
    x = stem_bn_->forward(x);
    x = stem_act_->forward(x);

    // Features (MBConv blocks)
    x = features_->forward(x);

    // Head
    x = head_conv_->forward(x);
    x = head_bn_->forward(x);
    x = head_act_->forward(x);
    x = avgpool_->forward(x);

    return x;
}

std::string EfficientNet::repr() const {
    std::ostringstream ss;
    ss << "EfficientNet(\n";
    ss << "  (stem_conv): " << stem_conv_->repr() << "\n";
    ss << "  (stem_bn): " << stem_bn_->repr() << "\n";
    ss << "  (stem_act): " << stem_act_->repr() << "\n";
    ss << "  (features): " << features_->repr() << "\n";
    ss << "  (head_conv): " << head_conv_->repr() << "\n";
    ss << "  (head_bn): " << head_bn_->repr() << "\n";
    ss << "  (head_act): " << head_act_->repr() << "\n";
    ss << "  (avgpool): " << avgpool_->repr() << "\n";
    if (classifier_) {
        ss << "  (classifier): " << classifier_->repr() << "\n";
    }
    ss << ")";
    return ss.str();
}

std::vector<nn::Parameter> EfficientNet::parameters() const {
    std::vector<nn::Parameter> params;

    auto add_params = [&params](const std::shared_ptr<nn::Module>& m, const std::string& prefix) {
        for (const auto& p : m->parameters()) {
            params.push_back({prefix + "." + p.name, p.spec, p.requires_grad});
        }
    };

    add_params(stem_conv_, "stem_conv");
    add_params(stem_bn_, "stem_bn");

    // Features
    for (size_t i = 0; i < features_->size(); ++i) {
        for (const auto& p : (*features_)[i]->parameters()) {
            params.push_back({"features." + std::to_string(i) + "." + p.name, p.spec, p.requires_grad});
        }
    }

    add_params(head_conv_, "head_conv");
    add_params(head_bn_, "head_bn");

    if (classifier_) {
        add_params(classifier_, "classifier");
    }

    return params;
}

std::map<std::string, nn::Parameter> EfficientNet::named_parameters(const std::string& prefix) const {
    std::map<std::string, nn::Parameter> params;
    std::string p = prefix.empty() ? "" : prefix + ".";

    for (const auto& param : parameters()) {
        params[p + param.name] = param;
    }

    return params;
}

std::vector<std::shared_ptr<nn::Module>> EfficientNet::children() const {
    std::vector<std::shared_ptr<nn::Module>> kids = {
        stem_conv_, stem_bn_, stem_act_,
        features_,
        head_conv_, head_bn_, head_act_, avgpool_
    };
    if (classifier_) {
        kids.push_back(classifier_);
    }
    return kids;
}

// ============================================================================
// Configuration Factory Functions
// ============================================================================

EfficientNetConfig efficientnet_b0_config() {
    return {
        "efficientnet_b0",
        1.0f,   // width_mult
        1.0f,   // depth_mult
        224,    // input_size
        0.2f,   // dropout
        EfficientNetConfig::default_block_configs()
    };
}

EfficientNetConfig efficientnet_b1_config() {
    return {
        "efficientnet_b1",
        1.0f,   // width_mult
        1.1f,   // depth_mult
        240,    // input_size
        0.2f,   // dropout
        EfficientNetConfig::default_block_configs()
    };
}

EfficientNetConfig efficientnet_b2_config() {
    return {
        "efficientnet_b2",
        1.1f,   // width_mult
        1.2f,   // depth_mult
        260,    // input_size
        0.3f,   // dropout
        EfficientNetConfig::default_block_configs()
    };
}

EfficientNetConfig efficientnet_b3_config() {
    return {
        "efficientnet_b3",
        1.2f,   // width_mult
        1.4f,   // depth_mult
        300,    // input_size
        0.3f,   // dropout
        EfficientNetConfig::default_block_configs()
    };
}

EfficientNetConfig efficientnet_b4_config() {
    return {
        "efficientnet_b4",
        1.4f,   // width_mult
        1.8f,   // depth_mult
        380,    // input_size
        0.4f,   // dropout
        EfficientNetConfig::default_block_configs()
    };
}

// ============================================================================
// Model Factory Functions
// ============================================================================

std::shared_ptr<EfficientNet> efficientnet_b0(int64_t num_classes, bool pretrained) {
    auto model = std::make_shared<EfficientNet>(efficientnet_b0_config(), num_classes);

    if (pretrained) {
        throw core::ConfigurationError("Pretrained weights not yet implemented");
    }

    return model;
}

std::shared_ptr<EfficientNet> efficientnet_b1(int64_t num_classes, bool pretrained) {
    auto model = std::make_shared<EfficientNet>(efficientnet_b1_config(), num_classes);

    if (pretrained) {
        throw core::ConfigurationError("Pretrained weights not yet implemented");
    }

    return model;
}

std::shared_ptr<EfficientNet> efficientnet_b2(int64_t num_classes, bool pretrained) {
    auto model = std::make_shared<EfficientNet>(efficientnet_b2_config(), num_classes);

    if (pretrained) {
        throw core::ConfigurationError("Pretrained weights not yet implemented");
    }

    return model;
}

std::shared_ptr<EfficientNet> efficientnet_b3(int64_t num_classes, bool pretrained) {
    auto model = std::make_shared<EfficientNet>(efficientnet_b3_config(), num_classes);

    if (pretrained) {
        throw core::ConfigurationError("Pretrained weights not yet implemented");
    }

    return model;
}

std::shared_ptr<EfficientNet> efficientnet_b4(int64_t num_classes, bool pretrained) {
    auto model = std::make_shared<EfficientNet>(efficientnet_b4_config(), num_classes);

    if (pretrained) {
        throw core::ConfigurationError("Pretrained weights not yet implemented");
    }

    return model;
}

}  // namespace pyflame_vision::models
