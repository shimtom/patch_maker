#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages


def main():
    description = 'patch_maker package'

    setup(
        name='patch_maker',
        version='0.1.0',
        author='shimtom',
        author_email='ii00zero1230@gmail.com',
        url='',
        description=description,
        long_description=description,
        zip_safe=False,
        include_package_data=True,
        packages=find_packages(),
        install_requires=[],
        tests_require=[],
        setup_requires=[],
    )


if __name__ == '__main__':
    main()
