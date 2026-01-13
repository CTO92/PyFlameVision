#pragma once

/// @file container.hpp
/// @brief Container modules (Sequential, ModuleList)
///
/// Container modules hold other modules and provide ways to compose them.

#include "pyflame_vision/nn/module.hpp"
#include <sstream>

namespace pyflame_vision::nn {

/// Sequential container - applies modules in order
///
/// Sequential chains modules together, passing the output of one
/// as the input to the next.
///
/// Example:
///   auto seq = Sequential({
///       std::make_shared<Conv2d>(3, 64, 3, 1, 1),
///       std::make_shared<BatchNorm2d>(64),
///       std::make_shared<ReLU>()
///   });
///   auto output = seq.forward(input);
///
/// @note Thread Safety: Sequential is thread-safe if all contained
///       modules are thread-safe.
class Sequential : public Module {
public:
    Sequential() = default;

    explicit Sequential(std::vector<std::shared_ptr<Module>> modules)
        : modules_(std::move(modules)) {}

    /// Apply all modules in sequence
    TensorSpec forward(const TensorSpec& input) const override {
        TensorSpec current = input;
        for (const auto& module : modules_) {
            current = module->forward(current);
        }
        return current;
    }

    std::string name() const override { return "Sequential"; }

    std::string repr() const override {
        std::ostringstream ss;
        ss << "Sequential(\n";
        for (size_t i = 0; i < modules_.size(); ++i) {
            ss << "  (" << i << "): " << modules_[i]->repr() << "\n";
        }
        ss << ")";
        return ss.str();
    }

    std::vector<std::shared_ptr<Module>> children() const override {
        return modules_;
    }

    std::map<std::string, Parameter> named_parameters(
        const std::string& prefix = ""
    ) const override {
        std::map<std::string, Parameter> result;
        for (size_t i = 0; i < modules_.size(); ++i) {
            std::string child_prefix = prefix.empty()
                ? std::to_string(i)
                : prefix + "." + std::to_string(i);
            auto child_params = modules_[i]->named_parameters(child_prefix);
            result.insert(child_params.begin(), child_params.end());
        }
        return result;
    }

    /// Add module to end of sequence
    void add(std::shared_ptr<Module> module) {
        modules_.push_back(std::move(module));
    }

    /// Get number of modules
    size_t size() const { return modules_.size(); }

    /// Check if empty
    bool empty() const { return modules_.empty(); }

    /// Access module by index
    /// @throws BoundsError if index out of range
    std::shared_ptr<Module> operator[](size_t index) const {
        if (index >= modules_.size()) {
            throw BoundsError(
                "Sequential index " + std::to_string(index) +
                " out of range (size=" + std::to_string(modules_.size()) + ")"
            );
        }
        return modules_[index];
    }

    /// Get module at index (same as operator[])
    std::shared_ptr<Module> get(size_t index) const {
        return (*this)[index];
    }

    /// Iterator support
    auto begin() const { return modules_.begin(); }
    auto end() const { return modules_.end(); }
    auto begin() { return modules_.begin(); }
    auto end() { return modules_.end(); }

private:
    std::vector<std::shared_ptr<Module>> modules_;
};

/// ModuleList - stores modules without defining forward pass
///
/// ModuleList is useful when you need to store a list of modules
/// but want to define how they're used in a subclass.
///
/// @note Thread Safety: ModuleList is thread-safe for read operations.
class ModuleList : public Module {
public:
    ModuleList() = default;

    explicit ModuleList(std::vector<std::shared_ptr<Module>> modules)
        : modules_(std::move(modules)) {}

    /// ModuleList doesn't define forward - subclasses must override
    TensorSpec forward(const TensorSpec& /*input*/) const override {
        throw ConfigurationError("ModuleList does not define forward()");
    }

    std::string name() const override { return "ModuleList"; }

    std::string repr() const override {
        std::ostringstream ss;
        ss << "ModuleList(\n";
        for (size_t i = 0; i < modules_.size(); ++i) {
            ss << "  (" << i << "): " << modules_[i]->repr() << "\n";
        }
        ss << ")";
        return ss.str();
    }

    std::vector<std::shared_ptr<Module>> children() const override {
        return modules_;
    }

    std::map<std::string, Parameter> named_parameters(
        const std::string& prefix = ""
    ) const override {
        std::map<std::string, Parameter> result;
        for (size_t i = 0; i < modules_.size(); ++i) {
            std::string child_prefix = prefix.empty()
                ? std::to_string(i)
                : prefix + "." + std::to_string(i);
            auto child_params = modules_[i]->named_parameters(child_prefix);
            result.insert(child_params.begin(), child_params.end());
        }
        return result;
    }

    /// Append module to list
    void append(std::shared_ptr<Module> module) {
        modules_.push_back(std::move(module));
    }

    /// Get number of modules
    size_t size() const { return modules_.size(); }

    /// Check if empty
    bool empty() const { return modules_.empty(); }

    /// Access module by index
    std::shared_ptr<Module> operator[](size_t index) const {
        if (index >= modules_.size()) {
            throw BoundsError(
                "ModuleList index " + std::to_string(index) +
                " out of range (size=" + std::to_string(modules_.size()) + ")"
            );
        }
        return modules_[index];
    }

    /// Iterator support
    auto begin() const { return modules_.begin(); }
    auto end() const { return modules_.end(); }

private:
    std::vector<std::shared_ptr<Module>> modules_;
};

}  // namespace pyflame_vision::nn
