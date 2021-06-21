#!/usr/bin/python3

from setuptools import setup, find_packages

with open("README.md", 'r') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as fh:
    install_requires = fh.read().splitlines()


print(find_packages())

setup(
    name='tdseird',
    version='0.0.1',
    author="Taarak Rapolu",
    author_email="taarak.rapolu@gmail.com",
    url="",
    packages=find_packages(),
    description='Time-dependent SEIRD Model',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        #"License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)",
        "Operating System :: OS Independent",
    ],
)
