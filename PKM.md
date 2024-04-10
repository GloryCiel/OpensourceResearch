tensorflow는 텍스트 분류 또한 가능하다.
그 예로 IMDB 데이터셋에 대한 이진 분류기를 학습시킬 수 있다.
```
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
```

tensorflow를 import하고 keras에서 layers와 losses도 동일하게 import해주었다.

IMDB데이터셋에는 인터넷 영화 데이터베이스에서 가져온 50000개의 영화 리뷰 텍스트가 포함되어있다.
절반을 나누어 학습용과 테스트용으로 사용하고, 각각의 데이터셋은 동일한 수로 긍정적,부정적 리뷰로 나뉘어져있다.
```
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
```
위 코드로 데이터셋을 다운로드하여 추출할 수 있다.

머신러닝 실험을 실행할 때 데이터세트를 train, validation 및 test의 세 부분으로 나누는 것이 가장 좋다.

IMDB 데이터세트는 이미 훈련과 테스트로 나누어져 있지만 검증 세트가 부족하다. 
아래 validation_split 인수를 사용하여 훈련 데이터를 80:20으로 분할하여 검증 세트를 생성한다.
```
batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', 
    seed=seed)
```
레이블은 0 또는 1이고 0은 부정적, 1은 긍정적 속성을 가진다.

검증을 위해 훈련 세트의 나머지 5000개 리뷰를 사용한다.
```
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    seed=seed)
```
다음에는 데이터를 표준화와, 토큰화, 벡터화하는 과정을 거쳐야하는데 표준화는 불필요한 요소를 제거, 토큰화는 문자열을 토큰단위로 분리, 벡터화는 이를 숫자로 변환하여 신경망과 호환되게 하는 것을 의미한다.
이때 텐서플로우에서 제공하는 tf.keras.layers.TextVectorization을 사용한다.
```
max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)
```
이 레이어를 사용하여 표준화, 토큰화, 벡터화를 수행한다.
sequence_length 상수를 주어 시퀀스를 정확히 맞출 수 있도록 한다.
```
# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)
```
위 과정은 adapt를 호출하여 문자열 인덱스를 정수로 빌드할 수 있도록 한다.
```
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)
```
최종 전처리 단계로 이전에 생성한 TextVectorization 레이어를 훈련, 검증 및 테스트 데이터세트에 적용한다.
```
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
```
위 코드를 사용하여 데이터셋으로 인한 메모리 병목현상을 방지할 수 있다.

이제 신경망을 생성한다.
```
embedding_dim = 16
model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

model.summary()
```
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, None, 16)          160016    
                                                                 
 dropout (Dropout)           (None, None, 16)          0         
                                                                 
 global_average_pooling1d (  (None, 16)                0         
 GlobalAveragePooling1D)                                         
                                                                 
 dropout_1 (Dropout)         (None, 16)                0         
                                                                 
 dense (Dense)               (None, 1)                 17        
                                                                 
=================================================================
Total params: 160033 (625.13 KB)
Trainable params: 160033 (625.13 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```
