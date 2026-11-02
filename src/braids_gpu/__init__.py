"""Explicitly selected CUDA backend for BRAIDS.

The production CPU solver is separate. Importing this package never initializes
CUDA, probes mesh size, or changes the selected execution backend.
"""
__version__='0.1.0-experimental'

