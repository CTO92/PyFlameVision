# PyFlameVision C++ Examples

add_executable(imagenet_transforms imagenet_transforms.cpp)
target_link_libraries(imagenet_transforms PRIVATE pyflame_vision)
target_include_directories(imagenet_transforms
    PRIVATE ${CMAKE_SOURCE_DIR}/include
)
