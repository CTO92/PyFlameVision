# python/CMakeLists.txt
# Python bindings for PyFlameVision

# ============================================================================
# Python extension module
# ============================================================================

pybind11_add_module(_pyflame_vision_cpp MODULE
    bindings.cpp
)

target_link_libraries(_pyflame_vision_cpp PRIVATE
    pyflame_vision
)

target_include_directories(_pyflame_vision_cpp PRIVATE
    ${PYFLAME_VISION_INCLUDE_DIR}
)

# Set output directory to match Python package structure
set_target_properties(_pyflame_vision_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/pyflame_vision
)

# ============================================================================
# Copy Python source files
# ============================================================================

file(GLOB_RECURSE PYTHON_SOURCE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/pyflame_vision/*.py"
)

foreach(PYTHON_FILE ${PYTHON_SOURCE_FILES})
    file(RELATIVE_PATH REL_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${PYTHON_FILE})
    configure_file(${PYTHON_FILE} ${CMAKE_BINARY_DIR}/python/${REL_PATH} COPYONLY)
endforeach()

# ============================================================================
# Generate _version.py
# ============================================================================

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/pyflame_vision/_version.py.in
    ${CMAKE_BINARY_DIR}/python/pyflame_vision/_version.py
    @ONLY
)

# ============================================================================
# Development install target
# ============================================================================

add_custom_target(pip-install-vision
    COMMAND ${Python3_EXECUTABLE} -m pip install -e ${CMAKE_BINARY_DIR}/python
    DEPENDS _pyflame_vision_cpp
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Installing PyFlameVision in development mode"
)
