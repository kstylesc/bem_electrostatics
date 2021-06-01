"""
bem_electrostatics
First cookiecutter try on Stefan's repository.
"""

# Add imports here
from .solute import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
