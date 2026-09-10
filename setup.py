from setuptools import find_packages
from distutils.core import setup

setup(
    name='elix_gym',
    version='1.0.0',
    author='Addverb Technologies',
    license="--",
    packages=find_packages(),
    author_email='humanoid@addverb.com',
    description='RL Isaac Gym environments for Elixis',
    install_requires=['isaacgym', 'rsl-rl', 'matplotlib']
)
