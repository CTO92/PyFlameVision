#pragma once

#define PYFLAME_VISION_VERSION_MAJOR 1
#define PYFLAME_VISION_VERSION_MINOR 0
#define PYFLAME_VISION_VERSION_PATCH 0
#define PYFLAME_VISION_VERSION_STRING "1.0.0-alpha"

namespace pyflame_vision {

inline constexpr int version_major = PYFLAME_VISION_VERSION_MAJOR;
inline constexpr int version_minor = PYFLAME_VISION_VERSION_MINOR;
inline constexpr int version_patch = PYFLAME_VISION_VERSION_PATCH;
inline constexpr const char* version_string = PYFLAME_VISION_VERSION_STRING;

}  // namespace pyflame_vision
