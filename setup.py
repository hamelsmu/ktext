from setuptools import setup
from setuptools import find_packages
from setuptools.command.install import install
import os
import ktext

class MyInstall(install):
    "custom addition to download spacy files"
    def run(self):
        os.system("python -m spacy download en")
        install.run(self)

setup(
    name='ktext',
    packages=find_packages(),
    version='0.40',
    description='Pre-processing text in parallel for Keras in python.',
    author='Hamel Husain',
    author_email='hamel.husain@gmail.com',
    url='https://github.com/hamelsmu/ktext',
    license='MIT',
    install_requires=['numpy',
                      'scipy',
                      'six',
                      'pandas>=0.21.0',
                      'pyyaml',
                      'pathos',
                      'msgpack',
                      'msgpack_numpy',
                      'dask',
                      'pyarrow',
                      'more_itertools',
                      'textacy==0.6.2',
                      'keras==2.2.4',
                      'tensorflow'],
    extras_require={
        'h5py': ['h5py'],
        'visualize': ['pydot'],
        'tests': ['pytest',
                  'pytest-pep8',
                  'pytest-xdist',
                  'pytest-cov',
                  'pandas'],
    },
    download_url='https://github.com/hamelsmu/ktext/archive/0.40.zip',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    cmdclass={'install': MyInstall}
)
