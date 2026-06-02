from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [l.strip() for l in f if l.strip() and not l.startswith("#")]

setup(
    name="logischain-ai",
    version="0.1.0",
    author="Zetheta Algorithms",
    author_email="research@zetheta.ai",
    description="Dual-domain AI system integrating supply chain intelligence into financial risk models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zetheta/logischain-ai",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Financial",
    ],
    entry_points={
        "console_scripts": [
            "logischain-demo=demo.app:main",
        ],
    },
)
