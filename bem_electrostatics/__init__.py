"""
bem_electrostatics
First cookiecutter try on Stefan's repository.
"""

# Add imports here
# Add imports here
import bempp.api
import os
from bem_electrostatics.solute import solute

BEM_ELECTROSTATICS_PATH = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
#WORKING_PATH = os.getcwd()

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
