from setuptools import setup, find_packages

setup(
    name='nltk_utils_lib',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'nltk',
        'numpy'
    ],
    description='Utility functions for NLP tasks using NLTK',
    author='Koyomi Hidaka',
    author_email='ftpxtf@mail.ru',
    url='https://github.com/KoyomiHidaka/NeuroAssist',  # URL проекта или документации
)

