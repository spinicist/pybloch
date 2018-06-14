"""Setup file to Not Another Neuroimaging Slicer

"""

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='Bloch',
    version='0.1.0',
    description='A collection of signal equations',
    author='Tobias Wood',
    author_email='tobias.wood@kcl.ac.uk',
    license='MPL',
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Neuroimagers',
                 'Topic :: Imaging',
                 'License :: OSI Approved :: Mozilla Public License',
                 'Proigramming Language :: Python :: 3',
    ],
    keywords='mri',
    packages=find_packages()
)