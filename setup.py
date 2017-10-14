from setuptools import find_packages, setup

setup(
    name='chainer2python3',
    version='',
    packages=find_packages(exclude=['images']),
    install_requires=['chainer>=2.0.1','numpy>=1.13.1', 'nibabel', 'GPy'],
    url='',
    license='',
    author='hashimoto',
    author_email='hashimoto@hal.t.u-tokyo.ac.jp',
    description=''
)
