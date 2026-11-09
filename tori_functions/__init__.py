"""Invariant tori of an Xsuite line, one by one, and their spectra. How to use them: the scripts in examples/."""
import os
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')        # GPUs numbered as in nvidia-smi
from .line import load
from .launch import launch_tori
from . import plots, plots4d
