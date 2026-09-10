# Week 5 — Deep Learning Application in Data Science

Task submitted for the **YuvaIntern Virtual Data Science with Python Trainee** internship (Week 5).

## Objective
Design, train, and evaluate a neural network on a publicly available dataset using
TensorFlow/Keras, with a critical analysis of challenges such as overfitting.

## Dataset and why it changed from previous weeks
Weeks 1-4 used the Palmer Penguins dataset, but that's small tabular data where a simple
Random Forest already hit 98%+ accuracy (Week 4) — not a meaningful problem for deep learning
to showcase its strengths on. This week switches to the **UCI Optical Recognition of Handwritten
Digits** dataset (1,797 images, 8x8 grayscale, 10 classes), accessed via
`sklearn.datasets.load_digits()`. This is genuine public image data, and was chosen specifically
because it could be loaded reliably offline (the standard MNIST dataset could not be downloaded
in this environment due to network restrictions — documented as a real resource constraint
encountered during the task).

## Repo structure
```
├── Week5_Deep_Learning_Report.docx   # Full report
├── data/
│   └── digits_split.npz              # Train/validation/test split (70/15/15, stratified)
├── model/
│   └── final_cnn_model.keras         # Trained final CNN
├── scripts/
│   ├── 01_data_prep.py               # Load, normalize, split, visualize sample digits
│   ├── 02_build_train.py             # Overfitting demo (Model A) + final regularized CNN (Model B)
│   ├── 03_evaluate_final.py          # Test set evaluation, confusion matrix, misclassifications
│   ├── 04_mlp_baseline.py            # Simple dense-network baseline for comparison
│   └── 05_architecture_diagram.py    # Generates the CNN architecture diagram
└── figures/                          # All generated chart images
```

## Key findings
- **Final CNN**: 99.63% test accuracy (only 1 misclassification out of 270 test images), using a
  compact 38,474-parameter architecture with BatchNorm, MaxPooling, and Dropout.
- **Overfitting demonstration**: an intentionally unregularized CNN (619,786 parameters, no
  dropout/augmentation) was trained alongside the final model for direct comparison. Its training
  loss collapsed toward zero while validation loss plateaued and crept up — a sign of overfitting
  visible in the loss curves before it clearly showed in accuracy.
- **MLP baseline comparison**: a simple dense network reached 98.52% accuracy, used to justify why
  a convolutional architecture was still the architecturally correct choice for image data.
- **Practical constraint**: standard MNIST could not be downloaded in this sandboxed environment
  (blocked network domain), so the task pragmatically switched to scikit-learn's offline-bundled
  digit dataset — documented in the report as a real resource-constraint decision.

## How to run
```bash
pip install tensorflow scikit-learn matplotlib seaborn numpy
cd scripts
python 01_data_prep.py
python 02_build_train.py
python 03_evaluate_final.py
python 04_mlp_baseline.py
python 05_architecture_diagram.py
```

## Full report
See [`Week5_Deep_Learning_Report.docx`](./Week5_Deep_Learning_Report.docx) for the complete
write-up.
