"""Data handling: reading-file parsing, metadata, preprocessing, masks, splits."""
from .masks import apply_region, merge_montgomery_masks, prepare_mask
from .metadata import build_metadata
from .preprocess import load_image01, preprocess_image, resize_with_pad
from .readings import parse_reading
from .splits import fold_assignment, load_folds, make_stratified_folds

__all__ = [
    "apply_region",
    "build_metadata",
    "fold_assignment",
    "load_folds",
    "load_image01",
    "make_stratified_folds",
    "merge_montgomery_masks",
    "parse_reading",
    "prepare_mask",
    "preprocess_image",
    "resize_with_pad",
]
