#!/usr/bin/env python

from setuptools import setup, find_packages

required = [
    'dm_control>=0.0.403778684', 'gym>=0.21.0', 'numpy>=1.22.1',
    'xmltodict>0.12.0'
]

setup(
    name='learn2learn_safely',
    version='0.0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=required)
