"""
Unit and regression test for the bwptdiag package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import bwptdiag


def test_bwptdiag_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "bwptdiag" in sys.modules
