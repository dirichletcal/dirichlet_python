import setuptools
from distutils.util import convert_path

with open("README.md", 'r') as f:
    long_description = f.read()

main_ns = {}
ver_path = convert_path('dirichletcal/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setuptools.setup(
    name='dirichletcal',
    version=main_ns['__version__'],
    author='Miquel Perello Nieto and Hao Song',
    author_email='perello.nieto@gmail.com',
    description='Python code for Dirichlet calibration',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dirichletcal/dirichlet_python',
    packages=setuptools.find_packages(),
    download_url = 'https://github.com/dirichletcal/dirichlet_python/archive/{}.tar.gz'.format(main_ns['__version__']),
    keywords = ['classifier', 'calibration', 'dirichlet', 'multiclass',
              'probability'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "jax",
        "jaxlib",
    ]
)
