import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name = "sen2tools", # Replace with your own username
    version = "0.0.1",
    author = "Olympia Gounari",
    author_email = "olympia_g@live.com",
    description = "Tools to manipulate Sentinel-2 satellite data",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url="https://github.com/Olyna/Sen2Tools",
    packages = setuptools.find_packages(),
    license="GNU General Public License v3 or later (GPLv3+)",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires = '>=3.6',
    install_requires=required,
)