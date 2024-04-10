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

가져온 데이터 중 필요하지 않은 부분(Time 등)을 제거한 뒤 학습에 사용한다. (해당 부분 코드 제외)

이후 데이터셋을 학습, 검증 그리고 테스트 셋으로 분할한다. 

검증 데이터셋은 모델에 적용하는 과정에서 손실을 평가하는데 사용되지만 모델에 직접 검증 데이터를 학습시키지는 않는다. 마찬가지로 테스트 셋 또한 학습된 모델을 평가하기 위해서만 사용된다.

이러한 과정은 일반적인 모델링에도 유용하나, 학습 데이터 부족으로 인해 오버피팅이 크게 우려되는 불균형 데이터 셋에서 특히 중요하게 여겨진다.

```
# sklearn에서 유틸리티를 가져와 데이터셋을 분할하고 섞는 데 사용한다.
train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# 레이블과 피쳐를 넘파이 배열로 바꿔준다.
train_labels = np.array(train_df.pop('Class'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('Class'))
test_labels = np.array(test_df.pop('Class'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)
```

이후 시각화를 위해 sklearn의 StandardScaler를 사용하여 입력 피쳐를 정규화한다. 정규화를 통해 평균은 0으로, 표준편차는 1로 설정한다.

```
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)
```

이후 부정 거래와 정상 거래 데이터의 분포를 비교하여 데이터 셋에 대해 확인한다.

다음의 코드를 통해 정규화 된 데이터의 분포를 확인할 수 있다.

```
pos_df = pd.DataFrame(train_features[ bool_train_labels], columns=train_df.columns)
neg_df = pd.DataFrame(train_features[~bool_train_labels], columns=train_df.columns)

sns.jointplot(x=pos_df['V5'], y=pos_df['V6'],
              kind='hex', xlim=(-5,5), ylim=(-5,5))
plt.suptitle("Positive distribution")

sns.jointplot(x=neg_df['V5'], y=neg_df['V6'],
              kind='hex', xlim=(-5,5), ylim=(-5,5))
_ = plt.suptitle("Negative distribution")
```

![Positive distribution](https://github.com/GloryCiel/OpensourceResearch/assets/63404135/7a7e4790-a386-4ea2-a7b3-8a35b0241caa)
![Negative distribution](https://github.com/GloryCiel/OpensourceResearch/assets/63404135/00e3999e-31c3-46da-ac9e-5f6d4041be71)

