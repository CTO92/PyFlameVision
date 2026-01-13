#pragma once

/// @file color_jitter.hpp
/// @brief Color jitter transform for data augmentation
///
/// Provides ColorJitter transform equivalent to
/// torchvision.transforms.ColorJitter.

#include "pyflame_vision/transforms/random_transform.hpp"
#include <array>
#include <algorithm>
#include <sstream>

namespace pyflame_vision::transforms {

/// Color jitter transform
///
/// Randomly adjusts brightness, contrast, saturation, and hue of an image.
/// Equivalent to torchvision.transforms.ColorJitter
///
/// Each factor can be:
/// - A single float: jitter range is [max(0, 1-value), 1+value]
/// - For hue: jitter range is [-value, value] where value in [0, 0.5]
///
/// @note Thread Safety: Thread-safe via inherited mutex-protected RNG.
class ColorJitter : public RandomTransform {
public:
    /// Create color jitter transform with single values
    /// @param brightness Brightness jitter amount (creates range [max(0, 1-b), 1+b])
    /// @param contrast Contrast jitter amount (creates range [max(0, 1-c), 1+c])
    /// @param saturation Saturation jitter amount (creates range [max(0, 1-s), 1+s])
    /// @param hue Hue jitter amount (creates range [-h, h], must be in [0, 0.5])
    ColorJitter(
        float brightness = 0.0f,
        float contrast = 0.0f,
        float saturation = 0.0f,
        float hue = 0.0f
    )
        : brightness_(to_range(brightness, false))
        , contrast_(to_range(contrast, false))
        , saturation_(to_range(saturation, false))
        , hue_(to_range(hue, true))
    {
        validate_params();
    }

    /// Create color jitter transform with explicit ranges
    /// @param brightness Brightness factor range [min, max]
    /// @param contrast Contrast factor range [min, max]
    /// @param saturation Saturation factor range [min, max]
    /// @param hue Hue shift range [min, max] (values must be in [-0.5, 0.5])
    ColorJitter(
        std::pair<float, float> brightness,
        std::pair<float, float> contrast,
        std::pair<float, float> saturation,
        std::pair<float, float> hue
    )
        : brightness_(brightness)
        , contrast_(contrast)
        , saturation_(saturation)
        , hue_(hue)
    {
        validate_params();
    }

    std::vector<int64_t> get_output_shape(
        const std::vector<int64_t>& input_shape
    ) const override {
        validate_input(input_shape);

        // Generate random factors
        last_brightness_ = random_uniform(brightness_.first, brightness_.second);
        last_contrast_ = random_uniform(contrast_.first, contrast_.second);
        last_saturation_ = random_uniform(saturation_.first, saturation_.second);
        last_hue_ = random_uniform(hue_.first, hue_.second);

        // Generate random order using Fisher-Yates shuffle
        generate_random_order();

        // Shape unchanged
        return input_shape;
    }

    std::string name() const override { return "ColorJitter"; }

    std::string repr() const override {
        std::ostringstream ss;
        ss << "ColorJitter(";

        bool first = true;
        auto add_param = [&](const std::string& name, const std::pair<float, float>& range,
                             float neutral_min, float neutral_max) {
            if (std::abs(range.first - neutral_min) > 1e-6f ||
                std::abs(range.second - neutral_max) > 1e-6f) {
                if (!first) ss << ", ";
                ss << name << "=(" << range.first << ", " << range.second << ")";
                first = false;
            }
        };

        add_param("brightness", brightness_, 1.0f, 1.0f);
        add_param("contrast", contrast_, 1.0f, 1.0f);
        add_param("saturation", saturation_, 1.0f, 1.0f);

        if (std::abs(hue_.first) > 1e-6f || std::abs(hue_.second) > 1e-6f) {
            if (!first) ss << ", ";
            ss << "hue=(" << hue_.first << ", " << hue_.second << ")";
        }

        ss << ")";
        return ss.str();
    }

    /// Get jitter ranges
    std::pair<float, float> brightness() const { return brightness_; }
    std::pair<float, float> contrast() const { return contrast_; }
    std::pair<float, float> saturation() const { return saturation_; }
    std::pair<float, float> hue() const { return hue_; }

    /// Get last applied factors
    float last_brightness_factor() const { return last_brightness_; }
    float last_contrast_factor() const { return last_contrast_; }
    float last_saturation_factor() const { return last_saturation_; }
    float last_hue_factor() const { return last_hue_; }

    /// Get last transform order (0=B, 1=C, 2=S, 3=H)
    std::array<int, 4> last_order() const { return last_order_; }

private:
    std::pair<float, float> brightness_;
    std::pair<float, float> contrast_;
    std::pair<float, float> saturation_;
    std::pair<float, float> hue_;

    mutable float last_brightness_ = 1.0f;
    mutable float last_contrast_ = 1.0f;
    mutable float last_saturation_ = 1.0f;
    mutable float last_hue_ = 0.0f;
    mutable std::array<int, 4> last_order_ = {0, 1, 2, 3};

    void validate_params() const {
        // Validate brightness, contrast, saturation ranges
        core::validate_color_factor(brightness_.first, "brightness_min");
        core::validate_color_factor(brightness_.second, "brightness_max");
        core::validate_color_factor(contrast_.first, "contrast_min");
        core::validate_color_factor(contrast_.second, "contrast_max");
        core::validate_color_factor(saturation_.first, "saturation_min");
        core::validate_color_factor(saturation_.second, "saturation_max");

        // Validate hue range
        core::validate_hue_factor(hue_.first);
        core::validate_hue_factor(hue_.second);

        // Validate ranges are valid (min <= max)
        if (brightness_.first > brightness_.second) {
            throw ValidationError("brightness range invalid: min > max");
        }
        if (contrast_.first > contrast_.second) {
            throw ValidationError("contrast range invalid: min > max");
        }
        if (saturation_.first > saturation_.second) {
            throw ValidationError("saturation range invalid: min > max");
        }
        if (hue_.first > hue_.second) {
            throw ValidationError("hue range invalid: min > max");
        }
    }

    /// Convert single value to range
    static std::pair<float, float> to_range(float value, bool is_hue) {
        if (value == 0.0f) {
            return is_hue ? std::make_pair(0.0f, 0.0f) : std::make_pair(1.0f, 1.0f);
        }
        if (is_hue) {
            return {-value, value};
        }
        return {std::max(0.0f, 1.0f - value), 1.0f + value};
    }

    /// Generate random transform order using Fisher-Yates shuffle
    void generate_random_order() const {
        last_order_ = {0, 1, 2, 3};
        for (int i = 3; i > 0; --i) {
            int j = random_int(0, i);
            std::swap(last_order_[i], last_order_[j]);
        }
    }
};

}  // namespace pyflame_vision::transforms
