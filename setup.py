from distutils.core import setup
from distutils.util import convert_path

with open("README.md", 'r') as f:
    long_description = f.read()

main_ns = {}
ver_path = convert_path('dirichletcal/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
  name = 'dirichletcal',
  packages = ['dirichletcal'],
  version=main_ns['__version__'],
  description = 'Diary to create notebooks and store intermediate results and figures',
  author = 'Miquel Perello Nieto and Hao Song',
  author_email = 'perello.nieto@gmail.com',
  url = 'https://github.com/perellonieto/DiaryPy',
  download_url = 'https://github.com/dirichletcal/dirichlet_python/archive/{}.tar.gz'.format(main_ns['__version__']),
  keywords = ['classifier', 'calibration', 'dirichlet', 'multiclass',
              'probability'],
  classifiers = [],
  long_description=long_description
)
