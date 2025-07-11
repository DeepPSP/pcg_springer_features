"""
"""

from pathlib import Path

import setuptools

from pcg_springer_features import __version__

cwd = Path(__file__).parent.resolve()

long_description = (cwd / "README.md").read_text(encoding="utf-8")

extras = {}
extras["test"] = [
    "pre-commit",
    "pytest",
    "pytest-xdist",
    "pytest-cov",
]
extras["dev"] = extras["test"]


setuptools.setup(
    name="pcg_springer_features",
    version=__version__,
    author="DeepPSP",
    author_email="wenh06@gmail.com",
    license="MIT",
    description="Tools for Computation of Springer's Features from Phonocardiogram (PCG).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepPSP/pcg_springer_features",
    # project_urls={},
    packages=setuptools.find_packages(
        exclude=[
            "docs*",
            "test*",
        ]
    ),
    # package_data=,
    # entry_points=,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=(cwd / "requirements.txt").read_text(encoding="utf-8").splitlines(),
    extras_require=extras,
)
