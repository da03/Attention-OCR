from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['distance', 'tensorflow', 'numpy', 'six']

setup(
    name='attentionocr',
    url='https://github.com/emedvedev/attention-ocr',
    author_name='Ed Medvedev',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='''Optical character recognition model
    for Tensorflow based on Visual Attention.'''
)
