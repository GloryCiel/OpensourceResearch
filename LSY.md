## 불균형 데이터 분류

TensorFlow는 데이터 분류 기술을 지원하는데, 해당 기술을 통해 불균형 데이터세트에 대한 분석을 시행할 수 있다.
다음은 Kaggle에서 지원하는 신용카드 부정행위 탐지 데이터들을 통해 작업하는 예시이다.
```
import tensorflow as tf
from tensorflow import keras

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```

기본적인 데이터 분류를 위한 pandas, matplotlib와 모델 학습를 위해 keras를, 이후 ROC와 PR-AUC를 통한 평가를 위한 sklearn을 import한다.

Kaggle에서 가져온 데이터는 총 284,807건의 거래에서 492건의 부정거래를 포함하고 있다.
불균형 데이터 분류는 소수 클래스에 해당하는 학습할 샘플이 거의 없기 때문에 모델 학습에 데이터를 적용할 때 주의해야하며, 가능한 많은 샘플을 수집하고, 모델이 소수 클래스를 최대한 효과적으로 활용할 수 있도록 어떤 피쳐가 관련되어 있는지에 주의해야 한다.
따라서 해당 데이터를 사용하기 이전에 주어진 데이터의 분포를 확인하는 과정이 필요하다.
아래는 Kaggle의 데이터를 가져온 뒤, .describe() 메소드를 사용하여 각 피쳐에 대한 통계를 확인하는 과정이다.
```
file = tf.keras.utils
raw_df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
raw_df.head()
raw_df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V26', 'V27', 'V28', 'Amount', 'Class']].describe()
```

