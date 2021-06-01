"""
Unit and regression test for the bem_electrostatics package.
"""

# Import package, test suite, and other packages as needed
import bem_electrostatics
import pytest
import sys

def test_bem_electrostatics_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "bem_electrostatics" in sys.modules
