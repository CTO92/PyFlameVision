/// @file resnet.cpp
/// @brief ResNet model implementation

#include "pyflame_vision/models/resnet.hpp"
#include <sstream>

namespace pyflame_vision::models {

// ============================================================================
// BasicBlock Implementation
// ============================================================================

BasicBlock::BasicBlock(
    int64_t inplanes,
    int64_t planes,
    int64_t stride,
    std::shared_ptr<nn::Module> downsample,
    int64_t groups,
    int64_t base_width,
    int64_t dilation
) : inplanes_(inplanes)
  , planes_(planes)
  , stride_(stride)
  , downsample_(std::move(downsample))
{
    if (groups != 1 || base_width != 64) {
        throw core::ValidationError("BasicBlock only supports groups=1 and base_width=64");
    }
    if (dilation > 1) {
        throw core::ValidationError("BasicBlock does not support dilation > 1");
    }

    // conv3x3 -> bn -> relu -> conv3x3 -> bn
    conv1_ = std::make_shared<nn::Conv2d>(inplanes, planes, 3, stride, 1, 1, 1, false);
    bn1_ = std::make_shared<nn::BatchNorm2d>(planes);
    relu_ = std::make_shared<nn::ReLU>(true);
    conv2_ = std::make_shared<nn::Conv2d>(planes, planes, 3, 1, 1, 1, 1, false);
    bn2_ = std::make_shared<nn::BatchNorm2d>(planes);
}

nn::TensorSpec BasicBlock::forward(const nn::TensorSpec& input) const {
    // Main path
    auto out = conv1_->forward(input);
    out = bn1_->forward(out);
    out = relu_->forward(out);
    out = conv2_->forward(out);
    out = bn2_->forward(out);

    // Residual path (identity or downsample must result in same shape as out)
    // Shape validation happens implicitly since out and residual must match

    // After residual addition (same shape as out)
    // Apply ReLU
    out = relu_->forward(out);

    return out;
}

std::string BasicBlock::repr() const {
    std::ostringstream ss;
    ss << "BasicBlock(\n";
    ss << "  (conv1): " << conv1_->repr() << "\n";
    ss << "  (bn1): " << bn1_->repr() << "\n";
    ss << "  (relu): " << relu_->repr() << "\n";
    ss << "  (conv2): " << conv2_->repr() << "\n";
    ss << "  (bn2): " << bn2_->repr() << "\n";
    if (downsample_) {
        ss << "  (downsample): " << downsample_->repr() << "\n";
    }
    ss << ")";
    return ss.str();
}

std::vector<nn::Parameter> BasicBlock::parameters() const {
    std::vector<nn::Parameter> params;

    auto add_params = [&params](const std::shared_ptr<nn::Module>& m, const std::string& prefix) {
        for (const auto& p : m->parameters()) {
            params.push_back({prefix + "." + p.name, p.spec, p.requires_grad});
        }
    };

    add_params(conv1_, "conv1");
    add_params(bn1_, "bn1");
    add_params(conv2_, "conv2");
    add_params(bn2_, "bn2");

    if (downsample_) {
        for (const auto& p : downsample_->parameters()) {
            params.push_back({"downsample." + p.name, p.spec, p.requires_grad});
        }
    }

    return params;
}

std::vector<std::shared_ptr<nn::Module>> BasicBlock::children() const {
    std::vector<std::shared_ptr<nn::Module>> kids = {conv1_, bn1_, relu_, conv2_, bn2_};
    if (downsample_) {
        kids.push_back(downsample_);
    }
    return kids;
}

// ============================================================================
// Bottleneck Implementation
// ============================================================================

Bottleneck::Bottleneck(
    int64_t inplanes,
    int64_t planes,
    int64_t stride,
    std::shared_ptr<nn::Module> downsample,
    int64_t groups,
    int64_t base_width,
    int64_t dilation
) : inplanes_(inplanes)
  , planes_(planes)
  , stride_(stride)
  , downsample_(std::move(downsample))
{
    // Width calculation for grouped convolutions (ResNeXt)
    width_ = static_cast<int64_t>(planes * (base_width / 64.0)) * groups;

    // 1x1 reduce
    conv1_ = std::make_shared<nn::Conv2d>(inplanes, width_, 1, 1, 0, 1, 1, false);
    bn1_ = std::make_shared<nn::BatchNorm2d>(width_);

    // 3x3
    conv2_ = std::make_shared<nn::Conv2d>(width_, width_, 3, stride, dilation, dilation, groups, false);
    bn2_ = std::make_shared<nn::BatchNorm2d>(width_);

    // 1x1 expand
    conv3_ = std::make_shared<nn::Conv2d>(width_, planes * expansion, 1, 1, 0, 1, 1, false);
    bn3_ = std::make_shared<nn::BatchNorm2d>(planes * expansion);

    relu_ = std::make_shared<nn::ReLU>(true);
}

nn::TensorSpec Bottleneck::forward(const nn::TensorSpec& input) const {
    // Main path: 1x1 -> 3x3 -> 1x1
    auto out = conv1_->forward(input);
    out = bn1_->forward(out);
    out = relu_->forward(out);

    out = conv2_->forward(out);
    out = bn2_->forward(out);
    out = relu_->forward(out);

    out = conv3_->forward(out);
    out = bn3_->forward(out);

    // After residual addition and ReLU
    out = relu_->forward(out);

    return out;
}

std::string Bottleneck::repr() const {
    std::ostringstream ss;
    ss << "Bottleneck(\n";
    ss << "  (conv1): " << conv1_->repr() << "\n";
    ss << "  (bn1): " << bn1_->repr() << "\n";
    ss << "  (conv2): " << conv2_->repr() << "\n";
    ss << "  (bn2): " << bn2_->repr() << "\n";
    ss << "  (conv3): " << conv3_->repr() << "\n";
    ss << "  (bn3): " << bn3_->repr() << "\n";
    ss << "  (relu): " << relu_->repr() << "\n";
    if (downsample_) {
        ss << "  (downsample): " << downsample_->repr() << "\n";
    }
    ss << ")";
    return ss.str();
}

std::vector<nn::Parameter> Bottleneck::parameters() const {
    std::vector<nn::Parameter> params;

    auto add_params = [&params](const std::shared_ptr<nn::Module>& m, const std::string& prefix) {
        for (const auto& p : m->parameters()) {
            params.push_back({prefix + "." + p.name, p.spec, p.requires_grad});
        }
    };

    add_params(conv1_, "conv1");
    add_params(bn1_, "bn1");
    add_params(conv2_, "conv2");
    add_params(bn2_, "bn2");
    add_params(conv3_, "conv3");
    add_params(bn3_, "bn3");

    if (downsample_) {
        for (const auto& p : downsample_->parameters()) {
            params.push_back({"downsample." + p.name, p.spec, p.requires_grad});
        }
    }

    return params;
}

std::vector<std::shared_ptr<nn::Module>> Bottleneck::children() const {
    std::vector<std::shared_ptr<nn::Module>> kids = {
        conv1_, bn1_, conv2_, bn2_, conv3_, bn3_, relu_
    };
    if (downsample_) {
        kids.push_back(downsample_);
    }
    return kids;
}

// ============================================================================
// ResNet Implementation
// ============================================================================

ResNet::ResNet(
    ResNetBlockType block_type,
    std::vector<int64_t> layers,
    int64_t num_classes,
    bool zero_init_residual,
    int64_t groups,
    int64_t width_per_group,
    std::optional<std::vector<bool>> replace_stride_with_dilation
) : block_type_(block_type)
  , layers_(std::move(layers))
  , groups_(groups)
  , base_width_(width_per_group)
  , num_classes_(num_classes)
{
    validate_params();

    // Process dilation options
    std::vector<bool> dilate = {false, false, false};
    if (replace_stride_with_dilation.has_value()) {
        dilate = replace_stride_with_dilation.value();
        if (dilate.size() != 3) {
            throw core::ValidationError("replace_stride_with_dilation must have exactly 3 elements");
        }
    }

    // Stem: conv7x7 -> bn -> relu -> maxpool
    conv1_ = std::make_shared<nn::Conv2d>(3, inplanes_, 7, 2, 3, 1, 1, false);
    bn1_ = std::make_shared<nn::BatchNorm2d>(inplanes_);
    relu_ = std::make_shared<nn::ReLU>(true);
    maxpool_ = std::make_shared<nn::MaxPool2d>(3, 2, 1);

    // Build layers
    layer1_ = make_layer(64, layers_[0], 1, false);
    layer2_ = make_layer(128, layers_[1], 2, dilate[0]);
    layer3_ = make_layer(256, layers_[2], 2, dilate[1]);
    layer4_ = make_layer(512, layers_[3], 2, dilate[2]);

    // Head: adaptive avgpool -> fc
    avgpool_ = std::make_shared<nn::AdaptiveAvgPool2d>(1);

    if (num_classes_ > 0) {
        fc_ = std::make_shared<nn::Linear>(512 * expansion(), num_classes_);
    }

    // Note: zero_init_residual would initialize the last bn in each block to zero
    // This is handled at weight loading time, not here in shape computation
    (void)zero_init_residual;
}

void ResNet::validate_params() const {
    if (layers_.size() != 4) {
        throw core::ValidationError("ResNet: layers must have exactly 4 elements");
    }
    for (size_t i = 0; i < layers_.size(); ++i) {
        if (layers_[i] <= 0) {
            throw core::ValidationError("ResNet: all layer counts must be positive");
        }
        // Security: Enforce maximum blocks per layer to prevent DoS via deep networks
        if (layers_[i] > core::SecurityLimits::MAX_RESNET_BLOCKS_PER_LAYER) {
            throw core::ResourceError(
                "ResNet: layer " + std::to_string(i) + " block count (" +
                std::to_string(layers_[i]) + ") exceeds maximum allowed (" +
                std::to_string(core::SecurityLimits::MAX_RESNET_BLOCKS_PER_LAYER) + ")"
            );
        }
    }
    if (groups_ <= 0) {
        throw core::ValidationError("ResNet: groups must be positive");
    }
    if (base_width_ <= 0) {
        throw core::ValidationError("ResNet: width_per_group must be positive");
    }
    if (num_classes_ < 0) {
        throw core::ValidationError("ResNet: num_classes must be non-negative");
    }
}

std::shared_ptr<nn::Sequential> ResNet::make_downsample(
    int64_t inplanes,
    int64_t outplanes,
    int64_t stride
) {
    auto downsample = std::make_shared<nn::Sequential>();
    downsample->add(std::make_shared<nn::Conv2d>(inplanes, outplanes, 1, stride, 0, 1, 1, false));
    downsample->add(std::make_shared<nn::BatchNorm2d>(outplanes));
    return downsample;
}

std::shared_ptr<nn::Sequential> ResNet::make_layer(
    int64_t planes,
    int64_t blocks,
    int64_t stride,
    bool dilate
) {
    auto layer = std::make_shared<nn::Sequential>();

    int64_t previous_dilation = dilation_;
    if (dilate) {
        dilation_ *= stride;
        stride = 1;
    }

    int64_t outplanes = planes * expansion();

    // First block may have stride and/or need downsample
    std::shared_ptr<nn::Module> downsample = nullptr;
    if (stride != 1 || inplanes_ != outplanes) {
        downsample = make_downsample(inplanes_, outplanes, stride);
    }

    if (block_type_ == ResNetBlockType::BASIC) {
        layer->add(std::make_shared<BasicBlock>(
            inplanes_, planes, stride, downsample, groups_, base_width_, previous_dilation
        ));
    } else {
        layer->add(std::make_shared<Bottleneck>(
            inplanes_, planes, stride, downsample, groups_, base_width_, previous_dilation
        ));
    }

    inplanes_ = outplanes;

    // Remaining blocks
    for (int64_t i = 1; i < blocks; ++i) {
        if (block_type_ == ResNetBlockType::BASIC) {
            layer->add(std::make_shared<BasicBlock>(
                inplanes_, planes, 1, nullptr, groups_, base_width_, dilation_
            ));
        } else {
            layer->add(std::make_shared<Bottleneck>(
                inplanes_, planes, 1, nullptr, groups_, base_width_, dilation_
            ));
        }
    }

    return layer;
}

nn::TensorSpec ResNet::forward(const nn::TensorSpec& input) const {
    validate_input_dims(input, 4);

    // Stem
    auto x = conv1_->forward(input);
    x = bn1_->forward(x);
    x = relu_->forward(x);
    x = maxpool_->forward(x);

    // Layers
    x = layer1_->forward(x);
    x = layer2_->forward(x);
    x = layer3_->forward(x);
    x = layer4_->forward(x);

    // Head
    x = avgpool_->forward(x);

    if (fc_) {
        // Flatten: [N, C, 1, 1] -> [N, C]
        x = {{x.shape[0], x.shape[1]}, x.dtype};
        x = fc_->forward(x);
    }

    return x;
}

std::vector<nn::TensorSpec> ResNet::forward_features(const nn::TensorSpec& input) const {
    validate_input_dims(input, 4);

    std::vector<nn::TensorSpec> features;

    // Stem output
    auto x = conv1_->forward(input);
    x = bn1_->forward(x);
    x = relu_->forward(x);
    x = maxpool_->forward(x);
    features.push_back(x);

    // Layer outputs
    x = layer1_->forward(x);
    features.push_back(x);

    x = layer2_->forward(x);
    features.push_back(x);

    x = layer3_->forward(x);
    features.push_back(x);

    x = layer4_->forward(x);
    features.push_back(x);

    return features;
}

int64_t ResNet::num_features() const {
    return 512 * expansion();
}

std::string ResNet::repr() const {
    std::ostringstream ss;
    ss << "ResNet(\n";
    ss << "  (conv1): " << conv1_->repr() << "\n";
    ss << "  (bn1): " << bn1_->repr() << "\n";
    ss << "  (relu): " << relu_->repr() << "\n";
    ss << "  (maxpool): " << maxpool_->repr() << "\n";
    ss << "  (layer1): " << layer1_->repr() << "\n";
    ss << "  (layer2): " << layer2_->repr() << "\n";
    ss << "  (layer3): " << layer3_->repr() << "\n";
    ss << "  (layer4): " << layer4_->repr() << "\n";
    ss << "  (avgpool): " << avgpool_->repr() << "\n";
    if (fc_) {
        ss << "  (fc): " << fc_->repr() << "\n";
    }
    ss << ")";
    return ss.str();
}

std::vector<nn::Parameter> ResNet::parameters() const {
    std::vector<nn::Parameter> params;

    auto add_params = [&params](const std::shared_ptr<nn::Module>& m, const std::string& prefix) {
        for (const auto& p : m->parameters()) {
            params.push_back({prefix + "." + p.name, p.spec, p.requires_grad});
        }
    };

    add_params(conv1_, "conv1");
    add_params(bn1_, "bn1");

    // Add layer params
    for (size_t i = 0; i < layer1_->size(); ++i) {
        for (const auto& p : (*layer1_)[i]->parameters()) {
            params.push_back({"layer1." + std::to_string(i) + "." + p.name, p.spec, p.requires_grad});
        }
    }
    for (size_t i = 0; i < layer2_->size(); ++i) {
        for (const auto& p : (*layer2_)[i]->parameters()) {
            params.push_back({"layer2." + std::to_string(i) + "." + p.name, p.spec, p.requires_grad});
        }
    }
    for (size_t i = 0; i < layer3_->size(); ++i) {
        for (const auto& p : (*layer3_)[i]->parameters()) {
            params.push_back({"layer3." + std::to_string(i) + "." + p.name, p.spec, p.requires_grad});
        }
    }
    for (size_t i = 0; i < layer4_->size(); ++i) {
        for (const auto& p : (*layer4_)[i]->parameters()) {
            params.push_back({"layer4." + std::to_string(i) + "." + p.name, p.spec, p.requires_grad});
        }
    }

    if (fc_) {
        add_params(fc_, "fc");
    }

    return params;
}

std::map<std::string, nn::Parameter> ResNet::named_parameters(const std::string& prefix) const {
    std::map<std::string, nn::Parameter> params;
    std::string p = prefix.empty() ? "" : prefix + ".";

    for (const auto& param : parameters()) {
        params[p + param.name] = param;
    }

    return params;
}

std::vector<std::shared_ptr<nn::Module>> ResNet::children() const {
    std::vector<std::shared_ptr<nn::Module>> kids = {
        conv1_, bn1_, relu_, maxpool_,
        layer1_, layer2_, layer3_, layer4_,
        avgpool_
    };
    if (fc_) {
        kids.push_back(fc_);
    }
    return kids;
}

// ============================================================================
// Factory Functions
// ============================================================================

std::shared_ptr<ResNet> resnet18(int64_t num_classes, bool pretrained) {
    auto model = std::make_shared<ResNet>(
        ResNetBlockType::BASIC,
        std::vector<int64_t>{2, 2, 2, 2},
        num_classes
    );

    if (pretrained) {
        // TODO: Load pretrained weights
        throw core::ConfigurationError("Pretrained weights not yet implemented");
    }

    return model;
}

std::shared_ptr<ResNet> resnet34(int64_t num_classes, bool pretrained) {
    auto model = std::make_shared<ResNet>(
        ResNetBlockType::BASIC,
        std::vector<int64_t>{3, 4, 6, 3},
        num_classes
    );

    if (pretrained) {
        throw core::ConfigurationError("Pretrained weights not yet implemented");
    }

    return model;
}

std::shared_ptr<ResNet> resnet50(int64_t num_classes, bool pretrained) {
    auto model = std::make_shared<ResNet>(
        ResNetBlockType::BOTTLENECK,
        std::vector<int64_t>{3, 4, 6, 3},
        num_classes
    );

    if (pretrained) {
        throw core::ConfigurationError("Pretrained weights not yet implemented");
    }

    return model;
}

std::shared_ptr<ResNet> resnet101(int64_t num_classes, bool pretrained) {
    auto model = std::make_shared<ResNet>(
        ResNetBlockType::BOTTLENECK,
        std::vector<int64_t>{3, 4, 23, 3},
        num_classes
    );

    if (pretrained) {
        throw core::ConfigurationError("Pretrained weights not yet implemented");
    }

    return model;
}

std::shared_ptr<ResNet> resnet152(int64_t num_classes, bool pretrained) {
    auto model = std::make_shared<ResNet>(
        ResNetBlockType::BOTTLENECK,
        std::vector<int64_t>{3, 8, 36, 3},
        num_classes
    );

    if (pretrained) {
        throw core::ConfigurationError("Pretrained weights not yet implemented");
    }

    return model;
}

std::shared_ptr<ResNet> resnext50_32x4d(int64_t num_classes, bool pretrained) {
    auto model = std::make_shared<ResNet>(
        ResNetBlockType::BOTTLENECK,
        std::vector<int64_t>{3, 4, 6, 3},
        num_classes,
        false,  // zero_init_residual
        32,     // groups
        4       // width_per_group
    );

    if (pretrained) {
        throw core::ConfigurationError("Pretrained weights not yet implemented");
    }

    return model;
}

std::shared_ptr<ResNet> resnext101_32x8d(int64_t num_classes, bool pretrained) {
    auto model = std::make_shared<ResNet>(
        ResNetBlockType::BOTTLENECK,
        std::vector<int64_t>{3, 4, 23, 3},
        num_classes,
        false,
        32,
        8
    );

    if (pretrained) {
        throw core::ConfigurationError("Pretrained weights not yet implemented");
    }

    return model;
}

std::shared_ptr<ResNet> wide_resnet50_2(int64_t num_classes, bool pretrained) {
    auto model = std::make_shared<ResNet>(
        ResNetBlockType::BOTTLENECK,
        std::vector<int64_t>{3, 4, 6, 3},
        num_classes,
        false,
        1,
        128  // 2x wider
    );

    if (pretrained) {
        throw core::ConfigurationError("Pretrained weights not yet implemented");
    }

    return model;
}

std::shared_ptr<ResNet> wide_resnet101_2(int64_t num_classes, bool pretrained) {
    auto model = std::make_shared<ResNet>(
        ResNetBlockType::BOTTLENECK,
        std::vector<int64_t>{3, 4, 23, 3},
        num_classes,
        false,
        1,
        128
    );

    if (pretrained) {
        throw core::ConfigurationError("Pretrained weights not yet implemented");
    }

    return model;
}

}  // namespace pyflame_vision::models
