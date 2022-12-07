# opengender

Simple gender SVM classifier trained on 600K names and gender labels.

Achieves `{'f1': 0.9091096133600594, 'accuracy': 0.908791969539633}` against the test set from [Comparison and benchmark of name-to-gender inference services](https://peerj.com/articles/cs-156/)


## How to train

0. Initialize environment

> poetry shell

1. Build the train and test splits

> python opengender/build.py

2. Train the model

> python opengender/train.py


## How to use

```py
from opengender import OpenGender

gender = OpenGender()

gender.predict('John')
```

> `{'gender': 'm', 'proba': 0.9709237188634068}`