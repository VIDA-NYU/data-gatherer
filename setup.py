from setuptools import setup, find_packages
import os

def read_readme():
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf8') as file:
        return file.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

long_description = read_readme()

setup(
    name='data_gatherer',
    version='0.1.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    python_requires='>=3.11',
    description="DataGatherer Library",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/VIDA-NYU/data-gatherer',
    keywords=['Information Extraction', 'NYU'],
)