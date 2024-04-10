tensorflow는 텍스트 분류 또한 가능하다.
그 예로 IMDB 데이터셋에 대한 이진 분류기를 학습시킬 수 있다.

import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

tensorflow를 import하고 keras에서 layers와 losses도 동일하게 import해주었다.

IMDB데이터셋에는 인터넷 영화 데이터베이스에서 가져온 50000개의 영화 리뷰 텍스트가 포함되어있다.
절반을 나누어 학습용과 테스트용으로 사용하고, 각각의 데이터셋은 동일한 수로 긍정적,부정적 리뷰로 나뉘어져있다.

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

위 코드로 데이터셋을 다운로드하여 추출할 수 있다.

머신러닝 실험을 실행할 때 데이터세트를 train, validation 및 test의 세 부분으로 나누는 것이 가장 좋다.

IMDB 데이터세트는 이미 훈련과 테스트로 나누어져 있지만 검증 세트가 부족하다. 
아래 validation_split 인수를 사용하여 훈련 데이터를 80:20으로 분할하여 검증 세트를 생성한다.

batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', 
    seed=seed)

레이블은 0 또는 1이고 0은 부정적, 1은 긍정적 속성을 가진다.

검증을 위해 훈련 세트의 나머지 5000개 리뷰를 사용한다.

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    seed=seed)

    
