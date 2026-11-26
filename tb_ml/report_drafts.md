# Draft text for the FPR

These are starting drafts for three sections of the report. **Read the cited
papers yourself, then rewrite these drafts in your own words.** Check your
university's policy on AI-assisted writing before you submit anything based on
them. Every number below comes from `results/` (seed 42), so re-check the
numbers if you change the code.

---

## A. Why these features and these models (Methodology)

*Fixes weakness 5. Add every paper you read to the Excel sheet, and extend
this section with them.*

**Features.** Before deep learning, computer-aided TB screening described each
radiograph with a small set of hand-crafted numerical features and passed them
to a standard classifier. Jaeger et al. (2014a) used intensity, gradient
(edge), shape and texture descriptors computed inside the segmented lung
fields, with a support vector machine. Their system was the first one evaluated
on the two public datasets used here (Jaeger et al., 2014b). This project
follows the same idea with a deliberately small set of 13 features in three
groups:

- **Intensity.** First-order statistics of the grey-level histogram: mean,
  standard deviation, skewness, kurtosis, entropy and median. TB opacities
  change how bright the lung fields are and how the grey values are spread.
- **Texture.** Five Haralick features from the grey-level co-occurrence matrix
  (GLCM): contrast, dissimilarity, homogeneity, energy and correlation
  (Haralick, Shanmugam and Dinstein, 1973). They describe how neighbouring
  pixels relate to each other. TB lesions such as consolidation, cavities and
  nodules make the lung texture patchier.
- **Edges.** The mean and standard deviation of the Sobel gradient magnitude.
  They measure how many sharp changes in brightness the image contains.

**Which models the literature uses.** The table summarises the classifiers
used with hand-crafted chest X-ray features on these datasets. Check each
entry against the full paper and add it to your Excel sheet.

| Paper | Features | Classifier | Datasets | Reported result |
|-------|----------|------------|----------|-----------------|
| Jaeger et al. (2014a) | intensity, edge, shape, texture inside segmented lungs | SVM | Montgomery, Shenzhen | AUC 0.87 / 0.90 |
| Santosh and Antani (2018) | multi-scale shape, edge and texture; left–right lung symmetry | voting of Bayesian network, MLP and random forest | Montgomery, Shenzhen (check the full text) | improves on earlier methods by >5% accuracy and 3% AUC |
| *Tuberculosis detection in chest radiographs using spotted hyena algorithm optimized deep and handcrafted features* (2022) – complete the authors from PMC9560840 | deep features plus hand-crafted texture | several, including decision tree ("fine tree") and SVM | Shenzhen (check) | 98.73% accuracy, 5-fold CV |

The SVM is the most established classifier for hand-crafted features on these
two datasets. Tree-based classifiers (decision trees and random forests)
appear in several later studies.

**Models.** Two classifiers were chosen because they are used with
hand-crafted features in the literature summarised above, and because they
work in very different ways:

- **Decision Tree.** A decision tree (Breiman et al., 1984) learns a sequence
  of if/then rules on single features. It is easy to interpret: the learned
  tree and its feature importances show which features drive the decision.
  However, it tends to overfit small datasets.
- **SVM.** A support vector machine with an RBF kernel (Cortes and Vapnik,
  1995) finds the maximum-margin boundary between the classes. It was the
  classifier used by Jaeger et al. (2014a), so it lets this project be compared
  with theirs.

**Evaluation.** Each model's hyperparameters were chosen by 5-fold
cross-validation inside the training data only. Performance was estimated with
repeated stratified 5-fold cross-validation (5 repeats, giving 25 test folds),
and the mean and standard deviation are reported. A single train/test split
was rejected because Montgomery's 138 images would leave only 28 test images,
and one misclassified image would change accuracy by about 4 percentage
points. Choosing the hyperparameters inside each training fold avoids the
optimistic bias that comes from tuning on the test data (Varma and Simon,
2006).

---

## B. Comparison with published results (Discussion)

*Fixes weakness 2.*

| | Montgomery AUC | Montgomery accuracy | Shenzhen AUC | Shenzhen accuracy |
|---|---|---|---|---|
| Jaeger et al. (2014a), SVM | 0.87 | 0.783 | 0.90 | 0.84 |
| This project, SVM (13 features) | 0.655 ± 0.076 | 0.616 ± 0.062 | 0.860 ± 0.038 | 0.784 ± 0.034 |
| This project, Decision Tree | 0.548 ± 0.080 | 0.561 ± 0.068 | 0.827 ± 0.033 | 0.772 ± 0.036 |

On Shenzhen, the SVM comes within about 0.04 AUC of Jaeger et al. (2014a),
using only 13 features computed from the whole image. On Montgomery the gap is
much larger, about 0.21 AUC. Three differences explain most of it:

1. **Lung segmentation.** Jaeger et al. first segmented the lungs and computed
   their features inside the lung fields only. This project computes its
   features from the whole image, so borders, text labels, the shoulders and
   the abdomen dilute the signal from the lungs.
2. **Number of features.** Jaeger et al. used a much larger set of descriptors,
   including shape and more texture and edge features, against 13 here.
3. **Dataset size.** Montgomery has only 138 images (58 TB), so each training
   fold has about 110. That is very few examples for learning a reliable
   boundary, and the wide spread across folds (± 0.08 AUC) reflects this.
   Shenzhen is almost five times larger, and the gap to the published result
   there is small.

The results should also be compared with care. The evaluation protocols
differ: here, repeated stratified 5-fold cross-validation with tuning inside
each fold. Check the exact protocol in Jaeger et al. (2014a) and describe it
here.

---

## C. Limitation: features from the whole image (Limitations)

*Fixes weakness 3.*

All 13 features are computed from the whole radiograph, not only from the lung
fields. The images from the two hospitals differ in ways that have nothing to
do with TB:

- Montgomery images are about 4020 × 4892 pixels with wide black borders.
- Shenzhen images are about 3000 × 3000 pixels.
- The two hospitals used different X-ray machines, exposure settings and
  burned-in text labels.

Global intensity and edge statistics pick up these differences directly. For
example, a wide black border lowers the mean intensity and adds a strong edge
whether or not the patient has TB.

The cross-dataset experiment (step 3) shows the consequence. A model trained on
one hospital and tested on the other usually collapses to predicting "Normal"
for almost every image:

- Shenzhen → Montgomery: the SVM's sensitivity is 0.00 and its AUC is 0.47.
- Montgomery → Shenzhen: the Decision Tree's sensitivity is 0.00.

What the model learned about "TB" on one hospital's images does not carry over
to the other hospital's images. Zech et al. (2018) reported the same problem
for a deep learning pneumonia model: its performance dropped on images from
hospitals it had not been trained on, and the model could identify which
hospital an image came from.

Restricting the features to the lung fields, as Jaeger et al. (2014a) did,
would remove part of this effect. Lung masks are available for Montgomery but
not for Shenzhen, so this is left as future work. The finding is still useful:
any TB screening tool built on global image statistics needs to be validated on
data from the hospital where it will be used.

---

## References

- Breiman, L., Friedman, J.H., Olshen, R.A. and Stone, C.J. (1984)
  *Classification and Regression Trees*. Belmont, CA: Wadsworth.
- Cortes, C. and Vapnik, V. (1995) 'Support-vector networks', *Machine
  Learning*, 20(3), pp. 273–297.
- Haralick, R.M., Shanmugam, K. and Dinstein, I. (1973) 'Textural features for
  image classification', *IEEE Transactions on Systems, Man, and Cybernetics*,
  SMC-3(6), pp. 610–621.
- Jaeger, S. et al. (2014a) 'Automatic tuberculosis screening using chest
  radiographs', *IEEE Transactions on Medical Imaging*, 33(2), pp. 233–245.
  doi:10.1109/TMI.2013.2284099.
- Jaeger, S. et al. (2014b) 'Two public chest X-ray datasets for computer-aided
  screening of pulmonary diseases', *Quantitative Imaging in Medicine and
  Surgery*, 4(6), pp. 475–477.
- Santosh, K.C. and Antani, S. (2018) 'Automated chest X-ray screening: can
  lung region symmetry help detect pulmonary abnormalities?', *IEEE
  Transactions on Medical Imaging*, 37(5), pp. 1168–1177.
  doi:10.1109/TMI.2017.2775636.
- Varma, S. and Simon, R. (2006) 'Bias in error estimation when using
  cross-validation for model selection', *BMC Bioinformatics*, 7, 91.
- Zech, J.R. et al. (2018) 'Variable generalization performance of a deep
  learning model to detect pneumonia in chest radiographs: a cross-sectional
  study', *PLOS Medicine*, 15(11), e1002683.
