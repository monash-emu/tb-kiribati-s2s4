"""Sphinx configuration: the summer2 to summer4 conversion guide (layout as summer4's docs/)."""

from __future__ import annotations

import os

project = "Kiribati TB on summer4"
copyright = "2026, Monash EMU"
author = "Monash EMU"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]
exclude_patterns = ["_build", "jupyter_execute", ".jupyter_cache", "Thumbs.db", ".DS_Store"]
master_doc = "index"
language = "en"

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "linkify"]
myst_heading_anchors = 3

# Notebooks are executed on every build, so the guide's numbers are the code's numbers and a
# broken cell fails the build.
nb_execution_mode = "cache"
nb_execution_timeout = 900
nb_execution_raise_on_error = True
nb_execution_cache_path = os.path.join(os.path.dirname(__file__), "_build", ".jupyter_cache")
nb_merge_streams = True

autodoc_default_options = {"members": True, "member-order": "bysource"}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

pygments_style = "friendly"
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "Kiribati TB: summer2 to summer4"
html_theme_options = {
    "navigation_depth": 2,
    "show_toc_level": 2,
    "logo": {"text": "Kiribati TB on summer4"},
    "icon_links": [
        {
            "name": "summer4",
            "url": "https://github.com/monash-emu/summer4",
            "icon": "fa-brands fa-github",
        }
    ],
}
html_context = {"default_mode": "light"}
