'''
Mini Project 1 COMP 472 AI
Authors: Dana Abousharbin and Jacques Sarr
'''
import numpy as np
from loadImages import load_images, load_labels
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
# Standard scientific Python 
import matplotlib.pyplot as plt


# Load training data -each has 60 000 images
X_train = load_images('train-images.idx3-ubyte') # rows of flattened images
y_train = load_labels('train-labels.idx1-ubyte') # correct numbers (0-9)

# Load testing data -each has 10 000 images
X_test = load_images('t10k-images.idx3-ubyte') # images
y_test = load_labels('t10k-labels.idx1-ubyte') # numbers

# normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\n       ---- Classification Report ----")
print(classification_report(y_test, y_pred))

print("\n       ---- Confusion Matrix ----")
print(confusion_matrix(y_test, y_pred))
