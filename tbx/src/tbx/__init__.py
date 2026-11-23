"""
tbx — cross-dataset tuberculosis screening on the NLM Montgomery and Shenzhen
chest X-ray sets.

The package is organised so that the *direction* of an experiment (which set
trains, which set tests) and the *image region* (whole image, lung-cropped,
lung-masked) are configuration settings rather than code changes. One training
and evaluation path runs every experiment; a YAML file names the rest.

Layout
------
tbx.config        experiment schema + YAML loader
tbx.data          reading-file parsing, metadata, preprocessing, masks, splits
tbx.eda           visual audit and distribution plots
tbx.segmentation  U-Net for generating Shenzhen lung masks
tbx.features      lung-field texture / intensity features (classical baseline)
tbx.models        classical (LR / SVM) and CNN (ResNet-18 / DenseNet-121)
tbx.evaluation    AUC, sensitivity / specificity, bootstrap CIs, DeLong's test
tbx.utils         seeding and small IO helpers
"""

__version__ = "0.1.0"
