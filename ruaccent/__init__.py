"""
RuAccent Predictor - Automatic stress accent prediction for Russian text.
"""

from .accentor import load_accentor, Accentor

__version__ = "1.2.0"
__author__ = "Eduard Emkuzhev"
__all__ = ["load_accentor", "Accentor"]