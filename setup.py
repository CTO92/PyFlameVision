#!/usr/bin/env python3
"""
PyFlameVision setup script with C++ extension building.

This script handles building the C++ extension module using CMake and pybind11.
For most installations, `pip install .` or `pip install pyflame-vision` will work.

For development:
    pip install -e .[dev]
"""

import os
import re
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """CMake extension module."""

    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    """Build extension using CMake."""

    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in sync with pybind11 module name in python/CMakeLists.txt
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Using this requires cmake>=3.18
        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake configuration arguments
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DPYFLAME_VISION_BUILD_PYTHON=ON",
            "-DPYFLAME_VISION_BUILD_TESTS=OFF",
            "-DPYFLAME_VISION_BUILD_EXAMPLES=OFF",
        ]

        # Check for PyFlame installation
        pyflame_dir = os.environ.get("PYFLAME_DIR")
        if pyflame_dir:
            cmake_args.append(f"-DPYFLAME_DIR={pyflame_dir}")
        else:
            # Check adjacent directory
            adjacent_pyflame = Path(ext.sourcedir).parent / "PyFlame"
            if adjacent_pyflame.exists():
                cmake_args.append(f"-DPYFLAME_DIR={adjacent_pyflame}")
            else:
                # Build in standalone mode
                cmake_args.append("-DPYFLAME_VISION_STANDALONE=ON")

        # Build arguments
        build_args = []

        # Windows-specific handling
        if sys.platform.startswith("win"):
            # Single config generators are handled by CMAKE_BUILD_TYPE above
            single_config = any(x in cmake_args for x in ("-A", "-G"))
            if not single_config:
                cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            build_args += ["--config", cfg]

        # macOS-specific handling
        if sys.platform == "darwin":
            # Cross-compile support for macOS
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += [f"-DCMAKE_OSX_ARCHITECTURES={';'.join(archs)}"]

        # Set parallel build based on environment or CPU count
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            import multiprocessing
            build_args += [f"-j{multiprocessing.cpu_count()}"]

        # Build directory
        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        # Run CMake configure
        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args],
            cwd=build_temp,
            check=True
        )

        # Run CMake build
        subprocess.run(
            ["cmake", "--build", ".", *build_args],
            cwd=build_temp,
            check=True
        )


def get_version() -> str:
    """Extract version from _version.py."""
    version_file = Path(__file__).parent / "python" / "pyflame_vision" / "_version.py"
    if version_file.exists():
        content = version_file.read_text()
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)
    return "1.0.0a1"  # Default version


# Only include CMake extension if building from source
ext_modules = []
if not os.environ.get("PYFLAME_VISION_PURE_PYTHON"):
    ext_modules = [CMakeExtension("pyflame_vision._pyflame_vision_cpp")]


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
