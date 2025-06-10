# COVID-19 Image Classifier

This project implements an image classification model using deep learning (TensorFlow/Keras) to classify chest X-ray images into one of the following three categories:

- **COVID-19**
- **Normal**
- **Viral Pneumonia**

The dataset used includes grayscale X-ray images organized into labeled folders. The model performs preprocessing, trains a fully connected neural network, and evaluates performance using accuracy, a confusion matrix, and a classification report.

---

## 🧠 Model Summary

- **Input**: Grayscale images resized to 150x150 pixels
- **Architecture**: Fully connected (`Dense`) layers with ReLU activation
- **Output Layer**: Softmax activation for 3-class prediction
- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy

---

## 📂 Dataset Structure

```
Covid19-dataset/
├── Covid/
├── Normal/
└── Viral Pneumonia/
```

Each folder contains chest X-ray images belonging to that class.

---

## 🖼️ Visualizations

- Displays random sample images from the dataset
- Shows model predictions vs actual labels with color-coded titles
- Displays a confusion matrix heatmap
- Prints a full classification report

---

## ⚙️ How to Run

1. Make sure your dataset is structured as shown above.
2. Install required libraries:

```bash
pip install numpy opencv-python matplotlib tensorflow scikit-learn seaborn tqdm
```

3. Run the Python file or open the notebook:

```bash
python covid_classifier.py
# OR
jupyter notebook covid_classifier.ipynb
```

---

## 📊 Evaluation

- Evaluates the model on test data
- Reports accuracy score
- Displays confusion matrix
- Outputs classification report (Precision, Recall, F1-score)

---

## 👩‍💻 Author

Created by [Your Name Here] – feel free to use, share, and improve it.
