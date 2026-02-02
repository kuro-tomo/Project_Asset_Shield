from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="shield",
    version="1.0.0",
    description="TIR Quantitative Trading Engine - Enterprise Grade",
    author="TIR Autonomous Squad",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=required,
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "shield-manage=shield.manage:main",
        ],
    },
)
