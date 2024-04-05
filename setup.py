from setuptools import setup

setup(
    name='my_illustris_python',
    version='1.0.0',
    packages=["my_illustris_python"],
    install_requires=["numpy", "h5py", "six"],
    tests_require=["nose","coverage"],
)
