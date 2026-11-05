"""W8A8 (INT8 weights per output channel, INT8 activations per token) linears for Ampere sm_86; see README.md."""
from .linear import W8A8Linear, W8A8Weight, gemm_quantised, w8a8_linear

__all__ = ["W8A8Linear", "W8A8Weight", "gemm_quantised", "w8a8_linear"]
