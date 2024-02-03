from setuptools import setup, find_packages

setup(
    name='rightmove_scraper',
    version='0.1',
    package_dir={'': 'src'},  # Tells setuptools that packages are under src
    packages=find_packages(where='src'),
)