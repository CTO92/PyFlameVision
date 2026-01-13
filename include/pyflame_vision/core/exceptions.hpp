#pragma once

#include <stdexcept>
#include <string>

namespace pyflame_vision {

/// Base exception for all PyFlameVision errors
class Exception : public std::runtime_error {
public:
    explicit Exception(const std::string& message)
        : std::runtime_error(message) {}
};

/// Thrown when input validation fails (invalid shapes, parameters, etc.)
class ValidationError : public Exception {
public:
    explicit ValidationError(const std::string& message)
        : Exception(message) {}
};

/// Thrown when array/container bounds are exceeded
class BoundsError : public Exception {
public:
    explicit BoundsError(const std::string& message)
        : Exception(message) {}
};

/// Thrown when integer overflow would occur
class OverflowError : public Exception {
public:
    explicit OverflowError(const std::string& message)
        : Exception(message) {}
};

/// Thrown when configuration is invalid
class ConfigurationError : public Exception {
public:
    explicit ConfigurationError(const std::string& message)
        : Exception(message) {}
};

/// Thrown when resource limits are exceeded
class ResourceError : public Exception {
public:
    explicit ResourceError(const std::string& message)
        : Exception(message) {}
};

/// Thrown when CSL template processing fails
class TemplateError : public Exception {
public:
    explicit TemplateError(const std::string& message)
        : Exception(message) {}
};

}  // namespace pyflame_vision
