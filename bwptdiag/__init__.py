"""Sample program exploring the relationship between standard, iterative diagonalization techniques and Brillouin-Wigner Perturbation Theory"""

# Add imports here
from .bwptdiag import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
