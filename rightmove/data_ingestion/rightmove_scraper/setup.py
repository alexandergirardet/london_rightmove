# Automatically created by: scrapyd-deploy

from setuptools import setup, find_packages

setup(
    name="rightmove_scraper",
    version="1.0",
    packages=find_packages(),
    entry_points={"scrapy": ["settings = rightmove_scraper.settings"]},
    package_data={"rightmove_scraper": ["resources/data/*.csv"]},
)
