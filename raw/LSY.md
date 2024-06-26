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

### 데이터 전처리

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


### 모델 학습 및 평가

모델은 다양한 방법으로 구축할 수 있다. 아래는 TensorFlow의 예제에서 제공한 모델 생성 함수이다.

```
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

def make_model(metrics=METRICS, output_bias=None):
  if output_bias is not None:
    output_bias = tf.keras.initializers.Constant(output_bias)
  model = keras.Sequential([
      keras.layers.Dense(
          16, activation='relu',
          input_shape=(train_features.shape[-1],)),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(1, activation='sigmoid',
                         bias_initializer=output_bias),
  ])

  model.compile(
      optimizer=keras.optimizers.Adam(learning_rate=1e-3),
      loss=keras.losses.BinaryCrossentropy(),
      metrics=metrics)

  return model
```

위와 같은 모델 생성 함수를 통해 다양한 학습 과정을 생성할 수 있다.

모델을 평가하기 위해서 가중치와 손실 등의 기준을 사용하는데, 다양한 학습 과정을 비교하기 위해 초기 모델의 가중치를 체크포인트 파일에 보관하고 학습 전에 각 모델에 로드하여 사용하는 편이 유용하다.

```
initial_weights = os.path.join(tempfile.mkdtemp(), 'initial_weights')
model.save_weights(initial_weights)

model = make_model()
model.load_weights(initial_weights)
```


모델을 비교하는 방법은 다양하지만 그 중 유용하게 사용하는 방식으로 ROC 곡선과 P-R 곡선(PR-AUC)가 있다.

두 곡선 모두 *sklearn.metrics* 에서 제공되어 쉽게 사용 가능하다. 아래는 해당 함수의 일반적인 사용법이며 plt에 그리는 방식은 일반적인 방식을 따른다.

```
def plot_roc(name, labels, predictions, **kwargs):
  fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)
```

일반적인 데이터 셋의 경우 위의 과정으로 생성한 그래프로 충분히 모델 학습이 가능하나, 불균형 데이터 셋의 경우 소수 클래스의 비율이 매우 부족하기에 소수 클래스에 대해 가중치를 두는 편이 모델 평가에 용이하다.

위의 모델 생성처럼 모델에 가중치를 주어 학습한 뒤 새로 ROC 곡선과 P-R 곡선을 그려 비교하면 그 차이는 다음과 같이 나타난다.

![weighted ROC](https://github.com/GloryCiel/OpensourceResearch/assets/63404135/552d5fe4-4898-4486-af93-0109076ed3ba)
![weighted PR AUC](https://github.com/GloryCiel/OpensourceResearch/assets/63404135/72ddf27f-5b0c-4328-adac-1e4849deefcc)
