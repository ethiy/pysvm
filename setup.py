#! /usr/bin/env python3
# -*- coding: <utf-8> -*-

from setuptools import setup, find_packages
from svm import __version__

def readme():
    """
    Longer description from readme.
    """
    with open('README.md', 'r') as readmefile:
        return readmefile.read()


def requirements():
    """
    Get requirements to install.
    """
    with open('requirements.txt', 'r') as requirefile:
        return [line.strip() for line in requirefile.readlines() if line]


setup(
    name='pysvm',
    version=__version__,
    description='Python SVM SMO',
    long_description=readme(),
    classifiers=[
        'License :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Research',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    platforms=[
        'Environment :: Console',
        'Operating System :: POSIX',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Operating System :: Windows'
    ],
    keywords='Machine Learning SVM SMO',
    url='https://github.com/ethiy/pysvm',
    author='Oussama Ennafii',
    author_email='oussama.ennafii@outlook.fr',
    license='License :: OSI Approved :: MIT License',
    packages=find_packages(exclude=['tests']),
    install_requires=requirements(),
    include_package_data=True,
    zip_safe=False
)