from setuptools import setup
from setuptools import find_packages


setup(
    name='kerasparalleltext',
    packages=find_packages(),
    version='0.12',
    description='Pre-processing text in parallel for Keras in python.',
    author='Hamel Husain',
    author_email='hamel.husain@gmail.com',
    url='https://github.com/hamelsmu/KerasParallelText',
    license='MIT',
    install_requires=['numpy',
                      'scipy',
                      'six',
                      'pandas'
                      'pyyaml',
                      'pathos',
                      'pyarrow',
                      'more_itertools',
                      'textacy',
                      'gensim',
                      'spacy',
                      'keras'],
    extras_require={
        'h5py': ['h5py'],
        'visualize': ['pydot'],
        'tests': ['pytest',
                  'pytest-pep8',
                  'pytest-xdist',
                  'pytest-cov',
                  'pandas'],
    },
    download_url='https://github.com/hamelsmu/KerasParallelText/archive/0.12.tar.gz',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)
