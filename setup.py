#!/usr/bin/env python

from setuptools import setup, find_packages

required = [
    'dm_control>=0.0.403778684', 'gym>=0.21.0', 'numpy>=1.22.1',
    'xmltodict>=0.12.0'
]

extras = {'dev': ['pytest>=4.4.0', 'Pillow', 'matplotlib']}

package_data = {'safe-adaptation-gym': ['assets/*']}

setup(
    name='safe-adaptation-gym',
    version='0.0.0',
    packages=find_packages(),
    python_requires='>3.8',
    include_package_data=True,
    install_requires=required,
    extras_require=extras,
    package_data=package_data)
