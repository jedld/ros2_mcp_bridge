from setuptools import find_packages, setup
import os
from glob import glob

package_name = "ros2_mcp_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.py")),
    ],
    install_requires=[
        "setuptools",
        "fastmcp>=2.0.0",
        "pyyaml",
        "numpy",
    ],
    zip_safe=True,
    maintainer="You",
    maintainer_email="you@example.com",
    description="MCP bridge for ROS 2 — exposes topics and services as LLM-callable tools.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "bridge = ros2_mcp_bridge.bridge:main",
        ],
    },
)
