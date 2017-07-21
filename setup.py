from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['distance', 'tensorflow', 'numpy', 'six']


def readme():
    with open('README.md') as file:
        return file.read()


setup(
    name='aocr',
    url='https://github.com/emedvedev/attention-ocr',
    author='Ed Medvedev',
    author_email='edward.medvedev@gmail.com',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    license='MIT',
    description='''Optical character recognition model
    for Tensorflow based on Visual Attention.''',
    long_description=readme(),
    entry_points={
        'console_scripts': ['aocr=aocr.launcher:main'],
    }
)
