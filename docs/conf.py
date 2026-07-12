project = 'KoCS'
copyright = '2026, 0010200303'
author = '0010200303'

extensions = [
    'breathe',
    'sphinx.ext.mathjax',
    'sphinx.ext.autodoc',
]

breathe_projects = {
    "KoCS": "./doxygen/xml"
}

breathe_default_project = "KoCS"

breathe_domain_by_extension = {
    "cpp": "cpp",
    "hpp": "cpp",
    "h": "cpp",
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = "furo"
html_static_path = ['_static']

breathe_show_define_links = False
