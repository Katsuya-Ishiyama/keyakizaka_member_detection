# -*- coding: utf-8 -*-

"""  MobileNetV2で欅坂46とけやき坂46のメンバーの顔認識 """

import keras
from keras.applications.mobilenetv2 import MobileNetV2
from sklearn.model_selection import train_test_split
from utils.utils import load_data


# Configuration for model
NUMBER_OF_MEMBERS = 41             # 漢字とひらがな合わせたメンバー数
CLASSES = NUMBER_OF_MEMBERS + 1    # one hot表現は0から始まるため
LOG_DIR = './logs'                 # LossとAccuracyのログ

# Configuration for learning
EPOCHS = 100
TEST_SIZE = 0.1
VALIDATION_SPLIT = 0.1


X, Y = load_data('/home/ishiyama/notebooks/keyakizaka_member_detection/image/mobilenet/')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, shuffle=True)


def execute_learning_mobilenetv2(alpha, depth_multiplier):
    model = MobileNetV2(alpha=alpha,
                        depth_multiplier=depth_multiplier,
                        include_top=True,
                        weights=None,
                        classes=CLASSES)
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    tensorboard = keras.callbacks.TensorBoard(log_dir=LOG_DIR+'/{}_{}'.format(alpha, depth_multiplier))
    csv_logger = keras.callbacks.CSVLogger('./csv/log_{}_{}.csv'.format(alpha, depth_multiplier))

    fit_result = model.fit(
        x=X_train,
        y=Y_train,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        verbose=2,
        callbacks=[tensorboard, csv_logger]
    )

    test_result = model.evaluate(
        x=X_test,
        y=Y_test
    )
    print('-----------------------------------------------')
    print('Test Result:')
    print('alpha = {}, depth_multiplier = {}'.format(alpha, depth_multiplier))
    print('Loss = {}, Accuracy = {}'.format(*test_result))

    model.save('keyakizaka_member_detection_mobilenetv2_{}_{}.h5'.format(alpha, depth_multiplier))


for alpha in [1.2, 1.1, 1, 0.9, 0.8]:
    for depth_multiplier in [0.8, 0.9, 1, 1.1, 1.2]:
        execute_learning_mobilenetv2(alpha, depth_multiplier)

