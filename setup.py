from setuptools import setup

setup(
    name='cardgame',
    version='0.0.0.52',
    packages=['cardgame'],
    url='',
    license='',
    author='Oleg Novokshonov',
    author_email='rainmaverick@mail.ru',
    install_requires=['setuptools>=51.0.0',
                      'tensorflow==2.3.0',
                      'numpy==1.18.5',
                      'pytz>=2018.9',
                      'matplotlib>=3.0.2',
                      'seaborn>=0.11.1',
                      'dataclasses>=0.6',
                      'pandas>=1.1.5',
                      'seaborn>=0.11.1',
                      'ipython>=5.5.0',
                      'scikit-learn>=0.22.2',
                      ]
    )