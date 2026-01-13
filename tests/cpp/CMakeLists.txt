# PyFlameVision C++ Tests

include(FetchContent)

# Fetch GoogleTest
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

# ImageTensor tests
add_executable(test_image_tensor
    test_image_tensor.cpp
)
target_link_libraries(test_image_tensor
    PRIVATE
        pyflame_vision
        GTest::gtest_main
)
target_include_directories(test_image_tensor
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Transform tests
add_executable(test_transforms
    test_transforms.cpp
)
target_link_libraries(test_transforms
    PRIVATE
        pyflame_vision
        GTest::gtest_main
)
target_include_directories(test_transforms
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Phase 3 transform tests
add_executable(test_phase3_transforms
    test_phase3_transforms.cpp
)
target_link_libraries(test_phase3_transforms
    PRIVATE
        pyflame_vision
        GTest::gtest_main
)
target_include_directories(test_phase3_transforms
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Phase 4 ops tests
add_executable(test_ops
    test_ops.cpp
)
target_link_libraries(test_ops
    PRIVATE
        pyflame_vision
        GTest::gtest_main
)
target_include_directories(test_ops
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(test_image_tensor)
gtest_discover_tests(test_transforms)
gtest_discover_tests(test_phase3_transforms)
gtest_discover_tests(test_ops)
