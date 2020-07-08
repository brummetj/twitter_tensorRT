

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup
    import os
    import re

    def find_packages(path=''):
        ret = []

        for root, dirs, files, in os.walk(path):
            if '__init__.py' in files:
                ret.append(re.sub('^[^A-z0-9_]+', '', root.replace('/', '.')))
        return ret

install_requires = [
    'numpy',
    'tensorflow-gpu==1.4.0',
    'tensorflow==1.15.2',
    'tweepy',
    'docopt',
    'pyspark'
]

setup(
    name='tweetRT',
    install_requires=install_requires,
    author='Joshua Brummet',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'tweetRT = tweetRT.run:main',
        ],
    },
)
