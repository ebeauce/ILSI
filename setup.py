"""
Minimal setup file for the ILSI library for Python packaging.
:copyright:
    Eric Beauce
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.en.html)
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ILSI",
    version="1.1.4",
    author="Eric Beaucé",
    author_email="ebeauce@mit.edu",
    description="Package for iterative linear stress inversion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ebeauce/ILSI",
    project_urls={
        "Bug Tracker": "https://github.com/ebeauce/ILSI/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL License",
        "Operating System :: OS Independent",
    ],
    license="GPL",
    package_dir={"": "."},
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "mplstereonet"],
    python_requires=">=3.6",
)
