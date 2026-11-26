"""Turn one chest X-ray into a short row of numbers (numerical features).

Every image becomes 13 numbers in three groups:

1. Intensity (first-order statistics) - how bright/dark the image is overall
   and how the grey values are spread out.
2. Texture (GLCM / Haralick features) - how neighbouring pixels relate to
   each other: smooth regions versus patchy, irregular regions. TB lesions
   (cavities, consolidation, nodules) make the lung texture patchier.
3. Edges (Sobel gradient) - how many sharp changes in brightness there are.

These are the kinds of hand-crafted features used in the TB screening
literature (e.g. Jaeger et al., 2014) before deep learning.
"""
import numpy as np
from PIL import Image
from scipy.stats import kurtosis, skew
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel

# Every image is resized to the same size so features are comparable.
IMAGE_SIZE = 256

# The GLCM counts pairs of grey levels. 256 levels gives a huge, sparse
# matrix, so the image is reduced to 32 grey levels first.
GLCM_LEVELS = 32

FEATURE_NAMES = [
    # intensity
    "mean", "std", "skewness", "kurtosis", "entropy", "median",
    # texture (GLCM)
    "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
    "glcm_energy", "glcm_correlation",
    # edges
    "edge_mean", "edge_std",
]

# The three groups, used in step 4 to see which kind of feature matters most.
FEATURE_GROUPS = {
    "intensity": FEATURE_NAMES[0:6],
    "texture": FEATURE_NAMES[6:11],
    "edges": FEATURE_NAMES[11:13],
    "all": FEATURE_NAMES,
}


def load_image(path):
    """Read a PNG as greyscale, resize to IMAGE_SIZE x IMAGE_SIZE, values 0-255."""
    img = Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.asarray(img, dtype=np.uint8)


def intensity_features(img):
    """Simple statistics of the pixel values."""
    pixels = img.ravel().astype(float)

    # Entropy: how unpredictable the grey values are (0 = one flat colour).
    counts = np.bincount(img.ravel(), minlength=256)
    p = counts / counts.sum()
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p))

    return {
        "mean": pixels.mean(),
        "std": pixels.std(),
        "skewness": skew(pixels),
        "kurtosis": kurtosis(pixels),
        "entropy": entropy,
        "median": np.median(pixels),
    }


def texture_features(img):
    """GLCM (grey-level co-occurrence matrix) texture features.

    The GLCM counts how often a pixel with grey level i sits next to a pixel
    with grey level j. We look one pixel away in four directions
    (0, 45, 90, 135 degrees) and average the result over the directions.
    """
    reduced = (img // (256 // GLCM_LEVELS)).astype(np.uint8)  # 0..31
    glcm = graycomatrix(
        reduced,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    feats = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        feats[f"glcm_{prop}"] = graycoprops(glcm, prop).mean()
    return feats


def edge_features(img):
    """Sobel filter: large values where brightness changes sharply."""
    edges = sobel(img.astype(float) / 255.0)
    return {"edge_mean": edges.mean(), "edge_std": edges.std()}


def extract_features(path):
    """All 13 features for one image, as a dict {feature_name: value}."""
    img = load_image(path)
    feats = {}
    feats.update(intensity_features(img))
    feats.update(texture_features(img))
    feats.update(edge_features(img))
    return {name: float(feats[name]) for name in FEATURE_NAMES}
