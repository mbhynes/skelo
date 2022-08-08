import sys
import os

sys.path.insert(0, os.path.abspath('../..'))

project = 'skelo'
copyright = '2022, Michael B Hynes'
author = 'Michael B Hynes'
from skelo import __version__ as release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.napoleon',
  'sphinx.ext.mathjax',
  'sphinx.ext.linkcode',
]
source_suffix = '.rst'

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

def linkcode_resolve(domain, info):
  if domain != 'py':
      return None
  if not info['module']:
      return None
  filename = info['module'].replace('.', '/')
  return f"https://github.com/mbhynes/skelo/tree/main/{filename}.py"
