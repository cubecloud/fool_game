import setuptools
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='fool_game',
    version='0.1.83',
    packages=['cardgames'],
    # name="example-pkg-YOUR-USERNAME-HERE", # Replace with your own username
    author='cubecloud',
    author_email='rainmaverick@mail.ru',
    description='The "fool" - card game with learned AI',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cubecloud/fool_game",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache 2.0 License",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": ""},
    python_requires=">=3.7",
    install_requires=['setuptools>=51.0.0',
                      'tensorflow==2.3.0',
                      'numpy==1.18.5',
                      'pytz>=2018.9',
                      'matplotlib>=3.0.2',
                      'seaborn>=0.11.1',
                      'dataclasses>=0.6',
                      'pandas>=1.1.5',
                      # 'ipython>=5.5.0',
                      ]
)
