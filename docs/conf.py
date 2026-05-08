project = 'KoCS'
copyright = '2026, 0010200303'
author = '0010200303'

extensions = [
  'breathe'
]

breathe_projects = {
  "KoCS": "./doxygen/xml"
}

breathe_default_project = "KoCS"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = "furo"
html_static_path = ['_static']
