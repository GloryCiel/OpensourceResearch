분산형 학습 기능

(Keras를 사용한 분산형 학습)

MirroredStrategy 개체를 생성한다. 이는 배포를 처리하고 내부에 모델을 구축하기 위한 컨텍스트 관리자를 제공한다.
strategy = tf.distribute.MirroredStrategy()


배치(batch) 크기와 학습률을 설정한다. 기본적으로 GPU 메모리에 맞춰 가능한 가장 큰 배치 크기로 설정한다.
num_train_examples = info.splits['train'].num_examples
num_test_examples = info.splits['test'].num_examples

BUFFER_SIZE = 10000

BATCH_SIZE_PER_REPLICA = 64
BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


특성 스케일링 함수 ( [0,255] 범위에서 [0,1] 범위로 이미지 픽셀 값을 정규화하는 함수)를 정의한다.
def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255

  return image, label


특성 스케일링 함수를 훈련 및 테스트 데이터에 적용한 후, 훈련 데이터를 섞고 이를 일괄 처리한다.
train_dataset = mnist_train.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
eval_dataset = mnist_test.map(scale).batch(BATCH_SIZE)


Strategy 컨텍스트 내에서 Keras API를 사용해 모델을 만들고 컴파일한다.
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


(Keras를 사용한 다중 작업자 학습)

각 작업자는 다른 시스템을 사용하기 때문에, 동일한 GPU를 사용하려고 하여 오류가 발생하지 않도록 모든 GPU를 비활성화한다. tf_config 환경 변수를 재설정하고, 현재 디렉터리가 Python의 경로에 있도록 하고, tf-nightly를 설치하고, tensorflow를 가져오는 설정을 한다.
import json
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG', None)
if '.' not in sys.path:
  sys.path.insert(0, '.')
pip install tf-nightly
import tensorflow as tf


몇 개의 epoch에 대해 모델을 훈련하고 단일 작업자의 결과를 관찰해 이상이 없는지 확인한다.
import mnist_setup

batch_size = 64
single_worker_dataset = mnist_setup.mnist_dataset(batch_size)
single_worker_model = mnist_setup.build_and_compile_cnn_model()
single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)


tensorflow에서 분산 훈련에는 여러 작업이 있는 cluster가 포함되며, 각 작업에는 하나 이상의 task가 있을 수 있다.
tf_config 구성 환경 변수는 각 작업자에 대한 cluster 구성을 지정하는 데 사용되는 JSON 문자열이다.
변수에는 cluster, task의 두 가지 구성 요소가 있다
: cluster는 모든 작업자에 대해 동일하며 worker 또는 chief와 같은 다양한 유형의 작업으로 구성된 사전인 훈련 클러스터에 대한 정보를 제공한다.
task는 현재 작업에 대한 정보를 제공하며 작업자마다 다르다. 이를 통해 해당 작업자의 type과 index가 지정된다.
다음은 구성의 예이다.
tf_config = {
    'cluster': {
        'worker': ['localhost:12345', 'localhost:23456']
    },
    'task': {'type': 'worker', 'index': 0}
}


모델을 훈련하려면 먼저 tf.distribute.MultiWorkerMirroredStrategy의 인스턴스를 만든다.
strategy = tf.distribute.MultiWorkerMirroredStrategy()


각 작업자가 실행할 main.py 파일은 다음과 같다.
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


global_batch_size = per_worker_batch_size * num_workers로 설정되어 각 작업자는 작업자의 수에 관계없이 per_worker_batch_size 만큼의 예제 배치를 처리하게 된다.


작업자는 tf_config를 JSON으로 직렬화하고 환경 변수에 추가한 후 main.py를 실행할 수 있다. 이후 로그 파일에 출력된 내용을 검사한다.
os.environ['TF_CONFIG'] = json.dumps(tf_config)

# first kill any previous runs
%killbgscripts

python main.py &> job_0.log

import time
time.sleep(10)

cat job_0.log


로그 파일의 마지막 줄이 Started server with target: grpc://localhost:12345 다음과 같으면 첫 번째 작업자가 준비되었으며, 다른 모든 작업자가 계속 진행할 준비가 되기를 기다린다.
다음 작업자 프로세스가 시작되도록 tf_config를 업데이트하고, 다음 작업자는 main.py를 실행시킨다.
tf_config['task']['index'] = 1
os.environ['TF_CONFIG'] = json.dumps(tf_config)

python main.py


모든 작업자가 활성 상태가 되면 훈련이 시작된다. 첫 번째 작업자의 로그를 확인하면 다른 작업자들도 모델 훈련에 참여했음을 알 수 있다.
cat job_0.log
