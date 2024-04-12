TensorFlow
======================

> 팀원
>  - 컴퓨터학부/2022110692/강혜원
>  - 컴퓨터학부/2023014038/박경민
>  - 컴퓨터학부/2022114435/이동훈
>  - 컴퓨터학부/2020111931/이시욱
>  - 컴퓨터학부/2020110353/장호진
 
목차
======================
> 1. 개요
> 2. 라이선스
> 3. 주요기능
>    - 신경 스타일 전이
>    - 텍스트 분류
>    - 분산형 학습 기능
>    - 불균형 데이터 분류
>    - 합성곱 신경망(Convolutional Neural Network, CNN)과 이미지 분류
****

# 1. 개요
TensorFlow는 Google에서 개발하고 지속적으로 관리하는 강력한 오픈소스 라이브러리이다. 이 라이브러리는 수치 계산, 대규모 머신 러닝, 딥러닝, 그리고 다양한 통계 및 예측 분석 워크로드를 지원하여, 데이터 획득부터 대규모 예측 및 미래 결과 개선까지의 전 과정을 개발자들에게 용이하게 만들어 준다. TensorFlow의 핵심 기능은 심층 신경망을 학습하고 실행하는 것이다. 개인용 컴퓨터의 CPU와 GPU에서 실행 가능하며, 높은 확장성과 유연성을 제공한다. 또한 TensorFlow는 Python을 프론트엔드 API로 사용하지만, C++, Java 등 다양한 언어의 래퍼를 지원하여 언어나 플랫폼에 관계없이 모델을 학습하고 배포할 수 있다. TensorFlow의 다국어로 번역된 공식 문서는 사용자가 빠르게 활용 방법을 익힐 수 있도록 도와준다.
****

# 2. 라이선스
TensorFlow의 라이선스는 아파치 2.0 오픈 소스 라이선스이다.
아파치 라이선스(Apache License)는 아파치 소프트웨어 재단에서 만든 소프트웨어 라이선스로, 대표적인 특징으로는 소스코드 공개의 의무가 없고 2차 라이선스와 변형물의 특허 출원이 가능하고, 라이선스 적용 시 아파치 재단의 이름과 라이선스의 내용을 명시해야 하며, 아파치 라이선스 2.0이 적용된 소스 코드를 수정했을 경우 외부에 그 사실을 밝혀야 한다.
그리고 특허 출원이 된 소스 코드의 사용자에게 특허의 무제한적 사용을 허가하여 개발자는 그 사용자에 대해 특허권 행사를 할 수 없다. 

배포시 의무사항은 다음과 같다.

수취인에게 라이선스 사본 제공

수정된 파일에 대해 수정사항을 표시한 안내문구 첨부

저작권, 특허, 상표, attribution에 대한 고지사항을 소스코드 또는 "NOTICE" 파일 등에 포함

최초개발자 등을 위해 보증을 면제하고, 책임을 제한

![image](https://github.com/GloryCiel/OpensourceResearch/assets/113595521/c8775e36-1108-48d8-ae18-3a326c0e9d12)

****

# 3. 주요기능
내용

## 3.1. 신경 스타일 전이

텐서플로우의 신경 스타일 전이(Neural Style Transfer)는 딥러닝 기술을 활용하여 한 이미지의 스타일(색감, 질감 등)을 다른 이미지에 적용하는 기술이다. 이를 통해 꽃 이미지에 반 고흐 풍의 스타일을 입히거나, 풍경 사진에 피카소 스타일을 적용하는 등의 결과물을 얻을 수 있다.

신경 스타일 전이는 다음과 같은 과정을 거친다:

콘텐츠(Content) 이미지와 스타일(Style) 이미지를 입력받는다.
콘텐츠 이미지를 보존하면서 스타일 이미지의 스타일 특징을 추출하는 손실 함수(Loss Function)를 정의한다.
입력 이미지를 전이하여 새로운 출력 이미지를 생성하고, 이 출력 이미지와 손실 함수 간의 차이를 최소화하는 방향으로 학습을 진행한다.
반복적인 최적화 과정을 통해 최종 출력 이미지를 생성한다. 이 이미지는 콘텐츠 이미지의 구조는 유지하면서 스타일 이미지의 색감과 질감을 반영한다.
이 과정에서 핵심적인 역할을 하는 것은 합성곱 신경망(Convolutional Neural Network, CNN)이다. 이 CNN은 이미지의 콘텐츠 특징과 스타일 특징을 추출하고, 그 차이를 최소화하도록 학습된다.

텐서플로우의 신경 스타일 전이 기능을 사용하려면 먼저 텐서플로우와 필요한 라이브러리를 설치해야 합니다. 아나콘다 환경에서 다음 명령을 실행하면 된다. 
```
conda install tensorflow
pip install numpy scipy matplotlib scikit-image
```



아래는 텐서플로우에서 제공하는 예제 코드를 일부 수정한 신경 스타일 전이를 구현한 코드 예시이다.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# 콘텐츠와 스타일 이미지 로드
content_image = plt.imread('path/to/content/image.jpg')
style_image = plt.imread('path/to/style/image.jpg')

# VGG19 모델 로드 (이미 학습된 모델 사용)
vgg_model = loadmat('vgg19_weights.mat')
vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

# 최적화를 위한 입력 이미지 생성 (노이즈로 초기화)
noise_image = np.random.uniform(0, 255, content_image.shape).astype('float32')

# 텐서 설정
content_tensor = tf.constant(content_image)
style_tensor = tf.constant(style_image)
noise_tensor = tf.Variable(tf.convert_to_tensor(noise_image))

# 손실 함수 정의 (콘텐츠 손실과 스타일 손실)
content_loss = tf.reduce_mean((content_tensor - vgg_model['conv4_2'][0][0][0]) ** 2)
style_loss = ... # 스타일 손실 계산 코드 생략

# 총 손실 함수
total_loss = content_loss * 1e-3 + style_loss * 1

# 최적화 단계 설정
optimizer = tf.train.AdamOptimizer(2.0)
train_step = optimizer.minimize(total_loss)

# 모델 학습
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(train_step)
    if i % 100 == 0:
        print(f'반복 횟수: {i}, 총 손실: {total_loss.eval(session=sess)}')

# 최종 이미지 결과 저장
final_image = noise_tensor.eval(session=sess)
plt.imsave('path/to/output/image.jpg', final_image)
```

이 코드는 먼저 콘텐츠 이미지와 스타일 이미지를 로드합다. 그리고 사전 학습된 VGG19 모델을 이용하여 이미지의 콘텐츠 특징과 스타일 특징을 추출한다. 

그 다음 노이즈로 초기화된 입력 이미지를 최적화하여, 콘텐츠 이미지의 구조와 스타일 이미지의 스타일 특징을 반영하도록 한다. 이를 위해 콘텐츠 손실과 스타일 손실을 정의하고, 두 손실의 가중 합을 최소화하는 방향으로 최적화를 진행한다.

1000번의 반복 학습 후, 최종 결과 이미지를 저장합니다. 이 이미지는 원래 콘텐츠 이미지의 구조는 유지하면서 스타일 이미지의 스타일 특징을 반영하게 된다.


## 3.2. 텍스트 분류

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
첫번째 레이어는 Embedding레이어로 정수로 인코딩된 리뷰를 입력받고 해당하는 임베딩 벡터를 찾는다.
두번째 GlobalAveragePooling1D 층은 sequence 차원에 대해 평균을 계산하여 각 샘플에 대해 고정된 길이의 출력 벡터를 반환한다.
세번째는 하나의 출력노드를 가진 연결층으로 sigmoid함수를 이용해 0과 1사이 값을 출력한다.

이제 손실함수와 옵티마이저를 준비해야한다. 
```
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
```
확률을 출력하는 이진분류이므로 binary_crossentropy를 사용한다.

dataset 개체를 fit 메서드에 전달하여 모델을 훈련시킨다.
```
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)
```
모델을 평가하기 위해 손실값과 정확도를 반환한다.
```
loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)
```
```
782/782 [==============================] - 3s 4ms/step - loss: 0.3108 - binary_accuracy: 0.8724
Loss:  0.31077080965042114
Accuracy:  0.8723599910736084
```
약 87%의 정확도임을 볼 수 있다.
model.fit()으로 History객체를 반환해 정확도와 손실그래프를 그릴 수 있다.
![image](https://github.com/GloryCiel/OpensourceResearch/assets/113595521/d300e3cd-8d0d-4540-9089-abe9523ccafe)
![image](https://github.com/GloryCiel/OpensourceResearch/assets/113595521/77e2f3c7-1962-4a4e-b93f-be3fcf561a50)
그래프에서 점선은 훈련의 손실과 정확도, 실선은 검증 손실과 검증 정확도이다.
검증 선의 결과는 epoch에 비례해 증가하지 않는데, 이는 과대적합 때문이다.
모델이 과도하게 최적화되어 일반화되지않은 데이터의 특징을 학습하기 때문이다.
텐서플로우 케라스에서 제공하는 콜백메서드를 활용해 검증정확도가 증가하지 않으면 학습을 중단할 수 있다.



## 3.3. 분산형 학습 기능

### Keras를 사용한 분산형 학습

`MirroredStrategy` 개체를 생성한다. 이는 배포를 처리하고 내부에 모델을 구축하기 위한 컨텍스트 관리자를 제공한다.
```python
strategy = tf.distribute.MirroredStrategy()
```


배치(batch) 크기와 학습률을 설정한다. 기본적으로 GPU 메모리에 맞춰 가능한 가장 큰 배치 크기로 설정한다.
```python
num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 10000

BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
```


특성 스케일링 함수 ( [0,255] 범위에서 [0,1] 범위로 이미지 픽셀 값을 정규화하는 함수)를 정의한다.
```python
def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255

  return image, label
```


특성 스케일링 함수를 훈련 및 테스트 데이터에 적용한 후, 훈련 데이터를 섞고 이를 일괄 처리한다.
```python
train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)
```


`Strategy` 컨텍스트 내에서 Keras API를 사용해 모델을 만들고 컴파일한다.
```python
with strategy.scope():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
  ])

  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])
```



### Keras를 사용한 다중 작업자 학습

각 작업자는 다른 시스템을 사용하기 때문에, 동일한 GPU를 사용하려고 하여 오류가 발생하지 않도록 모든 GPU를 비활성화한다.
`tf_config` 환경 변수를 재설정하고, 현재 디렉터리가 Python의 경로에 있도록 하고, `tf-nightly`를 설치하고, tensorflow를 가져오는 설정을 한다.
```python
import json
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG', None)
if '.' not in sys.path:
  sys.path.insert(0, '.')
pip install tf-nightly
import tensorflow as tf
```


몇 개의 epoch에 대해 모델을 훈련하고 단일 작업자의 결과를 관찰해 이상이 없는지 확인한다.
```python
import mnist_setup

batch_size = 64
single_worker_dataset = mnist_setup.mnist_dataset(batch_size)
single_worker_model = mnist_setup.build_and_compile_cnn_model()
single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)
```


tensorflow에서 분산 훈련에는 여러 작업이 있는 `cluster`가 포함되며, 각 작업에는 하나 이상의 `task`가 있을 수 있다.
`tf_config` 구성 환경 변수는 각 작업자에 대한 cluster 구성을 지정하는 데 사용되는 JSON 문자열이다.
변수에는 `cluster`, `task`의 두 가지 구성 요소가 있다
: `cluster`는 모든 작업자에 대해 동일하며 `worker` 또는 `chief`와 같은 다양한 유형의 작업으로 구성된 사전인 훈련 클러스터에 대한 정보를 제공한다.
task는 현재 작업에 대한 정보를 제공하며 작업자마다 다르다. 이를 통해 해당 작업자의 `type`과 `index`가 지정된다.
다음은 구성의 예이다.
```python
tf_config = {
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}
```


모델을 훈련하려면 먼저 `tf.distribute.MultiWorkerMirroredStrategy`의 인스턴스를 만든다.
```python
strategy = tf.distribute.MultiWorkerMirroredStrategy()
```


각 작업자가 실행할 main.py 파일은 다음과 같다.
```python
%%writefile main.py

import os
import json

import tensorflow as tf
import mnist_setup

per_worker_batch_size = 64
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

strategy = tf.distribute.MultiWorkerMirroredStrategy()

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_setup.mnist_dataset(global_batch_size)

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = mnist_setup.build_and_compile_cnn_model()


multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)
```

`global_batch_size = per_worker_batch_size * num_workers`로 설정되어 각 작업자는 작업자의 수에 관계없이 `per_worker_batch_size` 만큼의 예제 배치를 처리하게 된다.



작업자는 `tf_config`를 JSON으로 직렬화하고 환경 변수에 추가한 후 main.py를 실행할 수 있다. 이후 로그 파일에 출력된 내용을 검사한다.
```python
os.environ['TF_CONFIG'] = json.dumps(tf_config)

# first kill any previous runs
%killbgscripts

python main.py &> job_0.log

import time
time.sleep(10)

cat job_0.log
```


로그 파일의 마지막 줄이 `Started server with target: grpc://localhost:12345` 다음과 같으면 첫 번째 작업자가 준비되었으며, 다른 모든 작업자가 계속 진행할 준비가 되기를 기다린다.
다음 작업자 프로세스가 시작되도록 `tf_config`를 업데이트하고, 다음 작업자는 main.py를 실행시킨다.
```python
tf_config['task']['index'] = 1
os.environ['TF_CONFIG'] = json.dumps(tf_config)

python main.py
```


모든 작업자가 활성 상태가 되면 훈련이 시작된다. 첫 번째 작업자의 로그를 확인하면 다른 작업자들도 모델 훈련에 참여했음을 알 수 있다.
```python
cat job_0.log
```

## 3.4. 불균형 데이터 분류

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


## 3.5. 합성곱 신경망(Convolutional Neural Network, CNN)과 이미지 분류

 TensorFlow에서는 이미지 인식 및 패턴 인식과 관련된 작업에서 매우 효과적으로 사용되는 기법 중 하나인 합성곱 신경망 기능을 제공한다.
 이 기능은 입력 이미지를 여러 계층으로 구성된 필터로 스캔하여 특정 맵을 생성하고 이런 필터는 입력된 이미지의 특정 패턴이나 기능을 감지하는 데 사용된다.
 아래의 예제는 TensorFlow의 합성곱 신경망 기능을 응용하여 이미지 분류를 하고 그 학습 결과를 출력하는 예제이다.
 Tensorflow에서는 케라스 Sequential API를 사용하여 간단한 코드를 작성함으로써 모델을 만들고 학습시킬 수 있다.
모델 훈련에서 사용될 이미지는 [image pair.png]과 같이 단어와, 단어에 맞는 이미지 쌍으로 제공되고 훈련 이미지와 테스트를 위한 이미지로 구분되어진다.
우선 합성곱 신경망은 이미지의 배치된 크기를 무시하고 이미지의 형상에 대한
정보(image_height, image_width, color_channels)의 텐서를 사용한다.
여기에서 텐서는 벡터들의 집합으로 보자.
텐서를 사용하여 합성곱 층을 만드는데 이 때 입력으로는 CIFAR 이미지 형식인 형상 높이 32, 너비 32, RGB 3쌍의 값을 처리하는 정수 값 3이 정보로 주어진다.
구체적인 예제 코드는 아래와 같다.
```Python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
```

이 중 높이와 너비 차원은 학습이 진행될 수록 감소하는 경향이 있다. 
합성곱 층을 만든 결과는 [convolution layer.png]와 같다.
기본적인 합성곱 층이 완성 된 이후에는 Dense층을 추가하여야 하는데 Dense층은 위에서 구한 합성곱의 연산 결과를 하나 이상의 Dense 층에 입력으로 주어서 분류를 수행한다.
여기서의 주의할 점은 Dense층의 오기 전까지의 연산 결과는 차원이 3개이지만 입력으로 넣을 때에는 1차원으로 변환을 수행해야한다는 것이다. 변환을 수행한 이후의 결과는 [added dense.png]과 같다.Dense층 추가의 예제 코드는 아래와 같다.
```Python
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
```
이렇게 만들어진 텐서들을 컴파일 한 이후 실행한 결과를 학습 횟수에 따라 그래프로 나타낸 결과는 아래의 [conclusion.png]와 같다.
학습 이후 모델 평가를 위한 코드는 아래와 같다.
```Python
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
```
결과에 있어서 accuracy는 모델이 올바르게 데이터를 분류한 비율을 나타내며 val_accuracy는 검증 정확도라고 불리는 모델이 훈련 데이터를 기반으로 학습한 이후 검증 데이터를 사용하여 모델의 성능을 표현할 때 사용하는 지표이다.
두 accuracy 값이 학습을 진행함에 따라서 증가하는 추이를 보이고 있음을 알수있다.

<img src="image/JHJ/image pair.png" width="300" height="300">
<img src="image/JHJ/convolution layer.png" width="400" height="200"><img src="image/JHJ/added dense.png" width="400" height="200"><img src="image/JHJ/conclusion.png" width="400" height="200">

