from skbuild import setup

setup(
    name="mgard",
    version="1.5.0",
    description="MultiGrid Adaptive Reduction of Data (MGARD)",
    author="Brown University",
    license="Apache",
    packages=["mgard"],
    package_dir={"mgard": "python/mgard"},
    python_requires=">=3.7",
    cmake_args=["-DMGARD_ENABLE_PYTHON=ON", "-DBUILD_SHARED_LIBS=OFF"],
    cmake_process_manifest_hook=lambda manifest: [name for name in manifest if
                                                  name.endswith(".so")],
)
