import os
from pathlib import Path

from setuptools import setup

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
    Pybind11Extension = None
    build_ext = None


ROOT = Path(__file__).parent
SOURCE = ROOT / "rag_core.cpp"


def _build_extensions():
    if not SOURCE.exists() or Pybind11Extension is None:
        return []
    extra_compile_args = ["-std=c++17", "-O3", "-pthread"]
    if os.name == "nt":
        extra_compile_args = []
    return [
        Pybind11Extension(
            "rag_core",
            ["rag_core.cpp"],
            extra_compile_args=extra_compile_args,
        )
    ]


setup(
    name="rag-project",
    version="0.1.0",
    packages=["rag"],
    ext_modules=_build_extensions(),
    cmdclass={"build_ext": build_ext} if build_ext else {},
)
