from setuptools import find_packages, setup

DESCRIPTION = "IsoCoord: A Python library for computing isothermal coordinates using conformal mappings"

with open("README.md", "r", encoding="utf-8") as fh:
    LONG_DESCRIPTION = fh.read()

NAME = "isocoord"
AUTHOR = "Igor Bogush"
AUTHOR_EMAIL = "bogush94@gmail.com"
URL = "https://github.com/BogushPhysics/IsoCoord"
LICENSE = "MIT"
PYTHON_VERSION = ">=3.8, <3.14"

INSTALL_REQUIRES = [
    "numpy",
    "scipy",
]

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: MacOS",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
]

PLATFORMS = ["Linux", "Mac OSX", "Unix", "Windows"]
KEYWORDS = "conformal mapping, differential geometry, membranes, simulations, schrodinger, ginzburg-landau"

__version__ = "0.0.0"
exec(open("isocoord/version.py").read())

setup(
    name=NAME,
    version=__version__,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    license=LICENSE,
    packages=find_packages(),
    include_package_data=True,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords=KEYWORDS,
    classifiers=CLASSIFIERS,
    platforms=PLATFORMS,
    python_requires=PYTHON_VERSION,
    install_requires=INSTALL_REQUIRES,
    # extras_require=EXTRAS_REQUIRE,
)
