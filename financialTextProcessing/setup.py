from setuptools import setup

setup(
name='financialTextProcessing',
version='0.1',
description='Package for performing NLP and sentiment analysis on financial news',
url='https://github.com/jjacks95/BrainStationCapstone',
author='Joshua Jackson',
packages=['financialTextProcessing'],
install_requires=['pandas',
                'numpy',
                'matplotlib',
                'seaborn',
                'scikit-learn',
                'nltk',
                'spacy',
                'datetime',
                'yfinance',
                'requests',
                'backtrader',
                'tensorflow',
                ],
#tests_suite='nose.collector',
#tests_require=['nose']
)
