import sys
import os

sys.path.insert(0, os.path.abspath('../..'))

project = 'skelo'
copyright = '2022, Michael B Hynes'
author = 'Michael B Hynes'
release = "0.1"

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

autodoc_default_options = {
  'members': True,
  'member-order': 'bysource',
  'special-members': '__init__',
  'undoc-members': True,
  'exclude-members': '__weakref__',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

def linkcode_resolve(domain, info):
  if domain != 'py':
      return None
  if not info['module']:
      return None

  repo_base = 'https://github.com/mbhynes/skelo/tree/main'

  init_modules = [
    'skelo.model',
  ]

  if info['module'] in init_modules:
    filename = info['module'].replace('.', '/') + '/__init__.py'
  else:
    filename = info['module'].replace('.', '/') + '.py'
  return f"{repo_base}/{filename}"
