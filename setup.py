#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages


def main():
    description = 'patch maker package'

    setup(
        name='patch_maker',
        version='0.3.2',
        author='shimtom',
        author_email='ii00zero1230@gmail.com',
        url='',
        description=description,
        long_description=description,
        zip_safe=False,
        include_package_data=True,
        packages=find_packages(),
        install_requires=['numpy', 'pillow'],
        tests_require=[],
        setup_requires=[],
        entry_points={
            'console_scripts': [
                'makepatch = patch_maker.patch_maker:main'
            ]
        }
    )


if __name__ == '__main__':
    main()
