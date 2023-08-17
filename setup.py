from setuptools import setup

setup(
    name='tab_ddpm',
    version='0.1.0',    
    description='',
    url='https://github.com/R-N/tab-ddpm',
    author='',
    author_email='',
    license='',
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=1.10.1",
        "catboost~=1.0.3",
        "category-encoders~=2.3.0",
        "dython~=0.5.1",
        "icecream~=2.1.2",
        "libzero~=0.0.8",
        "numpy>=1.21.4",
        "optuna~=2.10.1",
        "pandas~=1.3.4",
        "pyarrow~=6.0.0",
        "rtdl~=0.0.9",
        "scikit-learn>=1.0.2",
        "scipy>=1.7.2",
        "skorch",
        "tomli-w~=0.4.0",
        "tomli~=1.2.2",
        "tqdm~=4.62.3",
        "delu",
        "rtdl",
        # smote
        "imbalanced-learn~=0.7.0",
        # tvae
        "rdt>=1.3.0",
    ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
    ],
)