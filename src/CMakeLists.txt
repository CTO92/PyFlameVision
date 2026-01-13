# src/CMakeLists.txt
# PyFlameVision library sources

# ============================================================================
# Source files
# ============================================================================

set(PYFLAME_VISION_TRANSFORM_SOURCES
    transforms/resize.cpp
    transforms/crop.cpp
    transforms/normalize.cpp
    transforms/compose.cpp
)

# Phase 2: Model architecture sources
set(PYFLAME_VISION_MODEL_SOURCES
    models/resnet.cpp
    models/efficientnet.cpp
)

# Backend sources (CSL code generation)
set(PYFLAME_VISION_BACKEND_SOURCES
    # These will be added as CSL generation is implemented
)

# ============================================================================
# Main library
# ============================================================================

add_library(pyflame_vision STATIC
    ${PYFLAME_VISION_TRANSFORM_SOURCES}
    ${PYFLAME_VISION_MODEL_SOURCES}
    ${PYFLAME_VISION_BACKEND_SOURCES}
)

target_include_directories(pyflame_vision
    PUBLIC
        $<BUILD_INTERFACE:${PYFLAME_VISION_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(pyflame_vision PUBLIC cxx_std_17)

set_target_properties(pyflame_vision PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME pyflame_vision
)

# Link PyFlame if available
if(NOT PYFLAME_VISION_STANDALONE AND PYFLAME_INCLUDE_DIR)
    target_include_directories(pyflame_vision PUBLIC ${PYFLAME_INCLUDE_DIR})
    target_compile_definitions(pyflame_vision PUBLIC PYFLAME_VISION_HAS_PYFLAME)
endif()

# ============================================================================
# CSL Templates
# ============================================================================

# Copy CSL templates to build directory
file(GLOB CSL_TEMPLATES "${CMAKE_CURRENT_SOURCE_DIR}/backend/csl_templates/*.csl.template")
if(CSL_TEMPLATES)
    file(COPY ${CSL_TEMPLATES} DESTINATION ${CMAKE_BINARY_DIR}/share/pyflame_vision/templates)
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS pyflame_vision
    EXPORT PyFlameVisionTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${PYFLAME_VISION_INCLUDE_DIR}/pyflame_vision
    DESTINATION include
)

if(CSL_TEMPLATES)
    install(FILES ${CSL_TEMPLATES}
        DESTINATION share/pyflame_vision/templates
    )
endif()
