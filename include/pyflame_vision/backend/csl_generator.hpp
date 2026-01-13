#pragma once

/// @file csl_generator.hpp
/// @brief Secure CSL template generator with parameter validation
///
/// This header provides secure template expansion for CSL code generation.
/// All parameters are validated before substitution to prevent code injection.
///
/// @note Thread Safety: All functions are thread-safe (no shared mutable state).

#include <string>
#include <unordered_map>
#include <sstream>
#include <fstream>
#include <regex>
#include <variant>
#include <optional>

#include "pyflame_vision/core/security.hpp"
#include "pyflame_vision/core/exceptions.hpp"

namespace pyflame_vision::backend {

/// Parameter types that can be substituted in CSL templates
enum class ParamType {
    NUMERIC,      // Integer or floating-point value
    IDENTIFIER,   // CSL identifier (variable/function name)
    DTYPE,        // CSL data type (f32, i32, etc.)
    BOOLEAN,      // true/false
    NUMERIC_ARRAY // Comma-separated numeric values
};

/// Validated template parameter
struct TemplateParam {
    std::string value;
    ParamType type;

    /// Create a validated numeric parameter
    static TemplateParam numeric(const std::string& val) {
        core::template_security::validate_numeric_value(val);
        return {val, ParamType::NUMERIC};
    }

    /// Create a validated numeric parameter from integer
    static TemplateParam numeric(int64_t val) {
        return {std::to_string(val), ParamType::NUMERIC};
    }

    /// Create a validated numeric parameter from float
    static TemplateParam numeric(float val) {
        if (!std::isfinite(val)) {
            throw core::TemplateError("Template numeric value must be finite");
        }
        return {std::to_string(val), ParamType::NUMERIC};
    }

    /// Create a validated identifier parameter
    static TemplateParam identifier(const std::string& val) {
        core::template_security::validate_identifier_value(val);
        return {val, ParamType::IDENTIFIER};
    }

    /// Create a validated dtype parameter
    static TemplateParam dtype(const std::string& val) {
        core::template_security::validate_dtype_value(val);
        return {val, ParamType::DTYPE};
    }

    /// Create a validated boolean parameter
    static TemplateParam boolean(bool val) {
        return {val ? "true" : "false", ParamType::BOOLEAN};
    }

    /// Create a validated numeric array parameter
    static TemplateParam numeric_array(const std::string& val) {
        core::template_security::validate_numeric_array(val);
        return {val, ParamType::NUMERIC_ARRAY};
    }
};

/// Secure CSL template generator
///
/// Expands template placeholders while validating all parameters
/// to prevent code injection attacks.
///
/// Template format:
/// - ${param_name} for parameter substitution
///
/// Example:
/// @code
/// CSLGenerator gen;
/// gen.set_param("width", TemplateParam::numeric(256));
/// gen.set_param("height", TemplateParam::numeric(256));
/// gen.set_param("dtype", TemplateParam::dtype("f32"));
/// std::string csl = gen.generate(template_content);
/// @endcode
class CSLGenerator {
public:
    using Params = std::unordered_map<std::string, TemplateParam>;

    CSLGenerator() = default;

    /// Load template content from file
    /// @param template_path Path to template file
    /// @return Template content as string
    /// @throws std::runtime_error if file cannot be opened
    static std::string load_template(const std::string& template_path) {
        std::ifstream file(template_path);
        if (!file) {
            throw std::runtime_error("Cannot open template: " + template_path);
        }
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }

    /// Set a template parameter
    /// @param name Parameter name (will be validated)
    /// @param param Validated parameter value
    void set_param(const std::string& name, TemplateParam param) {
        core::template_security::validate_param_name(name);
        params_[name] = std::move(param);
    }

    /// Set a numeric parameter (convenience overload)
    void set_numeric(const std::string& name, int64_t value) {
        set_param(name, TemplateParam::numeric(value));
    }

    /// Set a numeric parameter (convenience overload)
    void set_numeric(const std::string& name, float value) {
        set_param(name, TemplateParam::numeric(value));
    }

    /// Set an identifier parameter (convenience overload)
    void set_identifier(const std::string& name, const std::string& value) {
        set_param(name, TemplateParam::identifier(value));
    }

    /// Set a dtype parameter (convenience overload)
    void set_dtype(const std::string& name, const std::string& value) {
        set_param(name, TemplateParam::dtype(value));
    }

    /// Set a boolean parameter (convenience overload)
    void set_boolean(const std::string& name, bool value) {
        set_param(name, TemplateParam::boolean(value));
    }

    /// Clear all parameters
    void clear_params() {
        params_.clear();
    }

    /// Generate CSL code by substituting parameters
    /// @param template_content Template string with ${param} placeholders
    /// @return Generated CSL code
    /// @throws TemplateError if unsubstituted placeholders remain
    std::string generate(const std::string& template_content) const {
        std::string result = template_content;

        // Substitute all parameters
        for (const auto& [name, param] : params_) {
            std::string placeholder = "${" + name + "}";
            size_t pos = 0;
            while ((pos = result.find(placeholder, pos)) != std::string::npos) {
                result.replace(pos, placeholder.length(), param.value);
                pos += param.value.length();
            }
        }

        // Verify no unsubstituted placeholders remain
        verify_no_placeholders(result);

        return result;
    }

    /// Generate CSL code from file
    /// @param template_path Path to template file
    /// @return Generated CSL code
    std::string generate_from_file(const std::string& template_path) const {
        return generate(load_template(template_path));
    }

    /// Static method for one-shot generation with validation
    /// @param template_content Template string
    /// @param params Map of parameter name -> validated value
    /// @return Generated CSL code
    static std::string generate(
        const std::string& template_content,
        const Params& params
    ) {
        CSLGenerator gen;
        for (const auto& [name, param] : params) {
            gen.set_param(name, param);
        }
        return gen.generate(template_content);
    }

private:
    Params params_;

    /// Verify no unsubstituted placeholders remain
    static void verify_no_placeholders(const std::string& content) {
        // Regex to find ${...} patterns
        std::regex placeholder_regex(R"(\$\{([A-Za-z_][A-Za-z0-9_]*)\})");
        std::smatch match;
        if (std::regex_search(content, match, placeholder_regex)) {
            throw core::TemplateError(
                "Unsubstituted template placeholder: ${" + match[1].str() + "}"
            );
        }
    }
};

/// Builder pattern for CSL generator
///
/// Provides a fluent interface for building CSL code.
///
/// Example:
/// @code
/// std::string csl = CSLBuilder()
///     .set("width", 256)
///     .set("height", 256)
///     .set_dtype("dtype", "f32")
///     .set_bool("aligned", true)
///     .generate(template_content);
/// @endcode
class CSLBuilder {
public:
    /// Set numeric parameter
    CSLBuilder& set(const std::string& name, int64_t value) {
        gen_.set_numeric(name, value);
        return *this;
    }

    /// Set numeric parameter (float)
    CSLBuilder& set(const std::string& name, float value) {
        gen_.set_numeric(name, value);
        return *this;
    }

    /// Set identifier parameter
    CSLBuilder& set_id(const std::string& name, const std::string& value) {
        gen_.set_identifier(name, value);
        return *this;
    }

    /// Set dtype parameter
    CSLBuilder& set_dtype(const std::string& name, const std::string& value) {
        gen_.set_dtype(name, value);
        return *this;
    }

    /// Set boolean parameter
    CSLBuilder& set_bool(const std::string& name, bool value) {
        gen_.set_boolean(name, value);
        return *this;
    }

    /// Generate CSL code
    std::string generate(const std::string& template_content) const {
        return gen_.generate(template_content);
    }

    /// Generate CSL code from file
    std::string generate_from_file(const std::string& template_path) const {
        return gen_.generate_from_file(template_path);
    }

private:
    CSLGenerator gen_;
};

}  // namespace pyflame_vision::backend
