# TB detection in chest X-rays with numerical features and simple ML models

Can tuberculosis (TB) be detected in chest X-rays using a few numerical
features and simple machine learning models? This project tests that on two
public datasets from the US National Library of Medicine:

| Dataset | Hospital | Normal | TB | Total |
|---------|----------|--------|----|-------|
| Montgomery | Montgomery County, USA | 80 | 58 | 138 |
| Shenzhen | Shenzhen No.3 Hospital, China | 326 | 336 | 662 |

The project has four steps:

1. **Extract numerical features** from each X-ray: 13 numbers per image
   (`features.py`, `step1_extract_features.py`)
2. **Train two models** on the features, a **Decision Tree** and a **Support
   Vector Machine (SVM)**, and evaluate them on each dataset with repeated
   cross-validation (`step2_train_models.py`)
3. **Cross-dataset test**: train on one hospital and test on the other
   (`step3_cross_dataset.py`)
4. **Feature groups**: which kind of feature carries the TB signal?
   (`step4_feature_groups.py`)

Draft report text (choice of methods, comparison with Jaeger et al., and
limitations) is in [report_drafts.md](report_drafts.md).

## How to run

This folder is self-contained: its own Python environment (`.venv`), its own
data (`data/raw`) and its own outputs (`results/`).

```powershell
cd tb_ml
.\.venv\Scripts\Activate.ps1
python step1_extract_features.py
python step2_train_models.py
python step3_cross_dataset.py
python step4_feature_groups.py
```

Step 1 takes a few minutes. Steps 2 and 4 take a few minutes each, and step 3
under a minute. All outputs are written to `results/`.

**First time on a new computer:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python step0_download_data.py     # about 4 GB, only if data\raw is empty
```

The images live in `data\raw\MontgomerySet\CXR_png\` and
`data\raw\ChinaSet_AllFiles\CXR_png\`. The last digit of each filename is the
label: `_0` = normal, `_1` = TB.

If you move the images to `tb_ml\data\raw\`, you can drop the `--raw` argument.

## Step 1: the features

Each image is converted to greyscale and resized to 256 x 256 pixels. Then 13
numbers are computed from it:

| Group | Feature | What it measures |
|-------|---------|------------------|
| Intensity | mean, median | overall brightness |
| | std | spread of grey values (contrast) |
| | skewness | whether the histogram leans towards dark or bright |
| | kurtosis | whether the histogram is peaked or flat |
| | entropy | how varied and unpredictable the grey values are |
| Texture (GLCM) | contrast, dissimilarity | how different neighbouring pixels are |
| | homogeneity | how similar neighbouring pixels are |
| | energy | how uniform or repetitive the texture is |
| | correlation | how predictable a pixel is from its neighbour |
| Edges (Sobel) | edge_mean, edge_std | how many sharp brightness changes there are |

**GLCM (grey-level co-occurrence matrix).** The image is reduced to 32 grey
levels. The GLCM then counts how often grey level *i* sits next to grey level
*j*. This is done one pixel away in four directions (0°, 45°, 90° and 135°)
and averaged. The texture features are summaries of this matrix (Haralick et
al., 1973). TB lesions make the lung texture patchier, and that should change
these numbers.

`results/feature_boxplots.png` compares the value of every feature for normal
and TB images.

## Why these two models: the literature

Before deep learning, TB screening on these two datasets worked like this
project does: hand-made numerical features, then a standard classifier.

| Paper | Features | Classifier | Datasets |
|-------|----------|------------|----------|
| Jaeger et al. (2014), IEEE TMI | intensity, edge, shape, texture (inside lungs) | **SVM** | Montgomery, Shenzhen |
| Santosh & Antani (2018), IEEE TMI | shape, edge, texture | Bayes network + MLP + **random forest** (vote) | Montgomery, Shenzhen |
| *Spotted Hyena…* (2022), PMC9560840 | deep + hand-crafted (texture) | **decision tree** (fine tree), SVM, others | Shenzhen |

- The **SVM** is the classifier in the original paper on these datasets, so it
  gives a direct comparison.
- The **Decision Tree** appears in later TB work and was recommended by the
  supervisor. It is also the building block of the random forest. It is fully
  interpretable: you can draw the tree and read off its rules.

## Step 2: the models

**Decision Tree.** The tree learns a series of yes/no questions about the
features, for example "is glcm_contrast > 12.3?". It follows the answers down
to a leaf that says Normal or TB. At each split it picks the question that best
separates the two classes, which means the lowest *Gini impurity*. The setting
`max_depth` limits how many questions it can ask in a row, so it can't simply
memorise the training images. `results/tree_<dataset>.png` shows the learned
tree, and `tree_importance_<dataset>.png` shows which features it relied on.

**SVM (Support Vector Machine).** The SVM finds the boundary between Normal and
TB that leaves the widest gap (*margin*) to the nearest images of each class.
With the RBF kernel this boundary can be curved:

- `C` sets the trade-off between a wide margin and classifying every training
  image correctly.
- `gamma` sets how curved the boundary can be.

The SVM works with distances, so the features are first scaled to mean 0 and
standard deviation 1 (`StandardScaler`).

**Evaluation: repeated stratified 5-fold cross-validation.** Each dataset is
split into 5 parts with the same TB/normal ratio in each part (*stratified*).
Each part is the test set once, while the model trains on the other 4. This is
repeated 5 times with different shuffles, which gives **25 test results**. We
report their mean ± standard deviation. A single 80/20 split would leave
Montgomery with only 28 test images, where one wrong image moves accuracy by
about 4%. That is too unreliable.

**Choosing settings.** Inside each of the 25 folds, the model's settings are
chosen by a further 5-fold cross-validation **on that fold's training part
only**. The settings with the best average accuracy are kept. The test part is
never used to choose settings.

The confusion matrices and ROC curves use the first repeat, in which every
image is tested exactly once. The tree plot shows a tree trained on the whole
dataset, so you can see what it learns.

**Metrics.** TB is the positive class.

- **Accuracy**: share of images classified correctly
- **Sensitivity (recall)**: of the TB patients, how many were detected
- **Specificity**: of the healthy patients, how many were correctly cleared
- **Precision**: of the images flagged as TB, how many really were TB
- **F1**: a single number that balances precision and sensitivity
- **AUC**: the area under the ROC curve. It is the chance that a random TB
  image gets a higher TB score than a random normal image (0.5 = guessing,
  1.0 = perfect).

## Step 3: cross-dataset

The model is trained on all images from one hospital and tested on all images
from the other. The settings are again chosen by 5-fold cross-validation on the
training hospital. The drop compared with step 2 shows how well the models
cope with X-rays from a different hospital (different machines, patients and
image quality).

## Step 4: feature groups

Both models are trained on each feature group on its own (intensity, texture,
edges) and on all 13 features together, with the same repeated
cross-validation. This shows which kind of feature carries the TB signal.

## Results (seed 42)

Within each dataset (mean ± std over 25 test folds):

| Dataset | Model | Accuracy | Sensitivity | Specificity | F1 | AUC |
|---------|-------|----------|-------------|-------------|----|-----|
| Montgomery | Decision Tree | 0.561 ± 0.068 | 0.398 ± 0.210 | 0.680 ± 0.154 | 0.409 ± 0.144 | 0.548 ± 0.080 |
| Montgomery | SVM | 0.616 ± 0.062 | 0.411 ± 0.205 | 0.765 ± 0.147 | 0.440 ± 0.190 | 0.655 ± 0.076 |
| Shenzhen | Decision Tree | 0.772 ± 0.036 | 0.734 ± 0.063 | 0.812 ± 0.070 | 0.766 ± 0.039 | 0.827 ± 0.033 |
| Shenzhen | SVM | 0.784 ± 0.034 | 0.733 ± 0.050 | 0.837 ± 0.060 | 0.775 ± 0.036 | 0.860 ± 0.038 |

For comparison, Jaeger et al. (2014) reported AUC 0.87 on Montgomery and 0.90
on Shenzhen, with lung segmentation and a larger feature set. See
[report_drafts.md](report_drafts.md) for the discussion.

Feature groups (AUC, mean ± std over 25 folds):

| Dataset | Model | Intensity (6) | Texture (5) | Edges (2) | All (13) |
|---------|-------|---------------|-------------|-----------|----------|
| Montgomery | Decision Tree | 0.547 ± 0.088 | 0.513 ± 0.087 | 0.583 ± 0.099 | 0.548 ± 0.080 |
| Montgomery | SVM | 0.657 ± 0.096 | 0.602 ± 0.075 | 0.610 ± 0.085 | 0.655 ± 0.076 |
| Shenzhen | Decision Tree | 0.723 ± 0.045 | 0.819 ± 0.038 | 0.811 ± 0.041 | 0.827 ± 0.033 |
| Shenzhen | SVM | 0.806 ± 0.036 | 0.845 ± 0.037 | 0.814 ± 0.033 | 0.860 ± 0.038 |

Cross-dataset:

| Train → Test | Model | Accuracy | Sensitivity | Specificity | F1 | AUC |
|--------------|-------|----------|-------------|-------------|----|-----|
| Shenzhen → Montgomery | Decision Tree | 0.580 | 0.052 | 0.962 | 0.094 | 0.535 |
| Shenzhen → Montgomery | SVM | 0.580 | 0.000 | 1.000 | 0.000 | 0.468 |
| Montgomery → Shenzhen | Decision Tree | 0.492 | 0.000 | 1.000 | 0.000 | 0.557 |
| Montgomery → Shenzhen | SVM | 0.731 | 0.565 | 0.902 | 0.681 | 0.825 |

What these results show:

- Both models do well on Shenzhen. The SVM has the best AUC (0.86), within
  about 0.04 of Jaeger et al. (2014).
- Montgomery is much harder (best AUC 0.66). Each training fold has only about
  110 images, and results vary a lot between folds (± 0.08 AUC).
- On Shenzhen, texture alone (5 features) is almost as good as all 13
  features. On Montgomery, no feature group clearly stands out.
- The SVM beats the Decision Tree in every within-dataset comparison.
- Across hospitals, most models collapse to predicting "Normal" for almost
  every image. Features taken from the whole image pick up differences between
  the two hospitals' X-ray machines as well as differences caused by disease.

## Files

```
step0_download_data.py    downloads the two datasets into data/raw (first time only)
features.py               how one X-ray becomes 13 numbers
models.py                 Decision Tree + SVM, settings search, repeated CV, metrics
step1_extract_features.py builds results/features.csv
step2_train_models.py     within-dataset repeated 5-fold CV + plots
step3_cross_dataset.py    train on one hospital, test on the other
step4_feature_groups.py   intensity vs texture vs edges
report_drafts.md          draft report text: method choice, comparison, limitations
results/                  all outputs (CSV tables and PNG figures)
```

## References

- Haralick, R.M., Shanmugam, K. and Dinstein, I. (1973) 'Textural features for
  image classification', *IEEE Transactions on Systems, Man, and Cybernetics*,
  SMC-3(6), pp. 610–621.
- Jaeger, S. et al. (2014) 'Automatic tuberculosis screening using chest
  radiographs', *IEEE Transactions on Medical Imaging*, 33(2), pp. 233–245.
- Jaeger, S. et al. (2014) 'Two public chest X-ray datasets for computer-aided
  screening of pulmonary diseases', *Quantitative Imaging in Medicine and
  Surgery*, 4(6), pp. 475–477.
