from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='data-gatherer',
    version='0.1.0',
    author='Brandon Rose',
    author_email='author@example.com',
    description='Data gatherer for scientific articles',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/example/data-gatherer',
    packages=find_packages(include=['data_gatherer', 'data_gatherer.*']),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'pandas',
        'selenium',
        'cloudscraper',
        'lxml',
        'beautifulsoup4',
    ],
    entry_points={
        'console_scripts': [
            'data-gatherer=data_gatherer.cli:main',
        ],
    },
)
