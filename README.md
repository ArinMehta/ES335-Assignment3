# ES335-Assignment3

## Streamlit App
This is the link for the Streamlit app: [Streamlit App](https://es335-25-a3-q1-arin.streamlit.app/)

---

# 1. Next-Word Prediction using MLP (5 Marks)

In this question, you will extend the next-character prediction notebook (discussed in class) to a **next-word prediction problem**. That is, you will create an **MLP-based text generator**. You will train the model, visualize learned word embeddings, and finally deploy a **Streamlit app for interactive text generation**.

It is recommended to refer to **Andrej Karpathy’s blog post on the Effectiveness of RNNs**.

You must complete this task for **two datasets**:

* One from **Category I (Natural Language)**
* One from **Category II (Structured/Domain Text)**

---

## 1.1 Preprocessing and Vocabulary Construction (0.5 Marks)

For text-based datasets, you can remove special characters except **full stop (.)** so that it can be used to split sentences.

However, you **cannot ignore special characters** for other datasets like **C++ code**. In such cases, treat text between **newlines as a statement**.

To remove special characters from a line, you can use the following code snippet:

```python
import re
line = re.sub('[^a-zA-Z0-9 \.]', '', line)
```

This removes everything except:

* Alphanumeric characters
* Space
* Full stop (`.`)

Convert the text to **lowercase** and use **unique words to create the vocabulary**.

### Report

* Vocabulary size
* 10 most frequent words
* 10 least frequent words

To create **X and y pairs for training**, use a similar approach to the next-character prediction method.

---

## 1.2 Model Design and Training (1 Mark)

Build an **MLP-based text generator** with the following structure:

* **Embedding dimension:** 32 or 64
* **Hidden layers:** 1–2 layers (1024 neurons each)
* **Activation:** ReLU or Tanh
* **Output:** Softmax over vocabulary

Use **Google Colab or Kaggle** for training.

Training guidelines:

* Train for **500–1000 epochs**
* Start early since training takes time.

### Report in Notebook

* Training vs Validation Loss Plot
* Final Validation Loss / Accuracy
* Example Predictions
* Commentary on Learning Behavior

---

## 1.3 Embedding Visualization and Interpretation (1 Mark)

Visualize the learned embeddings using:

* **t-SNE** (if embedding dimension > 2)
* **Scatter Plot** (if embedding dimension = 2)

For visualization, select words such as:

* Synonyms
* Antonyms
* Names and pronouns
* Verbs and adverbs
* Words with no relations

### Discuss

* Clustering patterns
* Semantic relationships between words

---

## 1.4 Streamlit Application (1.5 Marks)

Write a **Streamlit application** that:

* Accepts **input text from the user**
* Predicts the **next k words or lines**

The Streamlit interface should include controls for:

* Context length
* Embedding dimension
* Activation function
* Random seed
* Temperature (to control randomness of predicted words)

You may use **any one of the datasets** used earlier.

You should also handle cases where **input words are not in the vocabulary**.

**Note:**

* No need to retrain the model from the app
* Train **2–3 model variants** and provide them as options.

---

## 1.5 Comparative Analysis (1 Mark)

Compare the **two trained models** (Category I vs Category II) based on:

### Dataset Characteristics

* Dataset size
* Vocabulary size
* Context predictability

### Model Performance

* Loss curves
* Quality of generated text

### Embedding Visualizations

### Summary

Provide insights on how **natural language differs from structured text in learnability**.

---

# Datasets

## Category I (Natural Language)

* Paul Graham Essays
* Wikipedia (English)
* Shakespeare
* *War and Peace* – Leo Tolstoy
* *The Adventures of Sherlock Holmes* – Arthur Conan Doyle

## Category II (Structured / Domain Text)

* Maths textbook
* Python or C++ code (Linux Kernel code)
* IITGN advisory generation
* IITGN website generation
* Generate sklearn documentation
* Notes generation
* Image generation (ASCII art, 0–255)
* Music generation
* Any comparable dataset (confirm with TA Neerja)

---

# 2. Moons Dataset & Regularization (3 Marks)

Generate the **Make-Moons dataset without using `sklearn.make_moons`**.

### Dataset Settings

* Default noise = **0.2**
* Additional test sets with noise = **0.1** and **0.3**
* Training set = **500 points**
* Test set = **500 points**

### Preprocessing

* Standardize features **after the train-test split**
* Use **train statistics only**
* Create **validation split (20%) from training set**
* Use **random seed = 1337**

---

## Train the Following Models

### 1. MLP with Early Stopping

* One hidden layer
* Early stopping with **patience = 50**

---

### 2. MLP with L1 Regularization

Grid search for:

```
λ ∈ {1e−6, 3e−6, 1e−5, 3e−5, 1e−4, 3e−4}
```

Report:

* Layer-wise sparsity
* Validation AUROC vs λ

---

### 3. MLP with L2 Regularization

* Tune penalty coefficient using validation dataset.

---

### 4. Logistic Regression with Polynomial Features

Examples:

* (x₁x₂)
* (x₁²)
* etc.

---

## Evaluation and Analysis

Evaluate:

* **Test accuracy** on noise = 0.20
* **Robustness accuracy** on noise = 0.10 and 0.30

Create a table including:

* Test accuracy for the **four models**
* Performance across **three noise levels**
* **Parameter count**

---

### Visualization

Plot **decision boundaries side-by-side** for all four models with noise = 0.2.

---

### Discussion

Discuss:

* Effect of **L1 regularization** on sparsity and boundary jaggedness
* Effect of **L2 regularization** on smoothness and margin

---

### Class Imbalance Experiment

Modify the **training set to 70:30 class imbalance**, while keeping the **test set balanced**.

Report:

* Accuracy
* AUROC

Discuss the **impact of class imbalance**.

---

# 3. MNIST and CNN Experiments (3 Marks)

This section explores **deep learning for images**.

You will:

* Train **MLPs and CNNs on MNIST**
* Compare performance against baseline models
* Visualize embeddings using **t-SNE**
* Test **cross-domain generalization on Fashion-MNIST**

---

# 3.1 Using MLP (1.5 Marks)

Train an **MLP on the MNIST dataset**.

Dataset:

* Training set: **60,000 images**
* Test set: **10,000 images**

If compute is limited, you may use a **stratified subset** of the training data but **keep the same test set**.

### MLP Architecture

* First layer: **30 neurons**
* Second layer: **20 neurons**
* Output layer: **10 neurons (for 10 classes)**

---

### Report

Compare against:

* Random Forest
* Logistic Regression

Metrics:

* Accuracy
* F1-score
* Confusion Matrix

Discuss:

* Observations
* Misclassifications

---

### t-SNE Visualization

Visualize the **20-neuron layer embeddings** for:

* **Trained model**
* **Untrained model**

Compare the two.

---

### Cross-Domain Testing

Test the trained MLP on **Fashion-MNIST**.

Discuss:

* Observations
* Compare **t-SNE plots** of MNIST vs Fashion-MNIST embeddings.

---

# 3.2 Using CNN (1.5 Marks)

Implement a **simple CNN** with:

* Convolution layer: **32 filters (3×3)**
* MaxPool layer
* Fully connected layer: **128 neurons**
* Output layer: **10 neurons**
* Activation: **ReLU**

Train on **MNIST**.

---

## Pretrained CNN Models

Additionally use **two pretrained CNNs** for inference, such as:

* AlexNet
* MobileNet
* EfficientNet

---

## Compare All Three Models

Metrics:

* Accuracy
* F1-score
* Confusion Matrix

Also report:

* Model size (number of parameters)
* Inference time on test set

---
