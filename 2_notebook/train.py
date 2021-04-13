# coding: utf-8

import glob
import os
import pickle
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import util
from common.gradient import numerical_gradient
from common.layers import (Affine, BatchNormalization, Convolution, Dropout,
                           MaxPooling, ReLU, SoftmaxWithLoss)
from common.models import CustomConvNet_withBatchNorm, TwoLayerNet
from common.optimizer import Adam, RMSProp


def onehot_to_str(label):
    """
    ワンホットベクトル形式のラベルをカタカナ文字に変換する
    """
    dic_katakana = {
        "a": 0,
        "i": 1,
        "u": 2,
        "e": 3,
        "o": 4,
        "ka": 5,
        "ki": 6,
        "ku": 7,
        "ke": 8,
        "ko": 9,
        "sa": 10,
        "si": 11,
        "su": 12,
        "se": 13,
        "so": 14,
    }
    label_int = np.argmax(label)
    for key, value in dic_katakana.items():
        if value == label_int:
            return key

def data_augumenter(train_data_path, train_label_path):
    X = []
    Y = []
    n = 6
    train_data = np.load(train_data_path)
    train_label = np.load(train_label_path)
    print("train_data.shape=", train_data.shape)
    print("train_label.shape=", train_label.shape)
    for i in range(len(train_data)):
        # 画像読み込み
        data = np.load(train_data_path)
        label = np.load(train_label_path)
        data = data[i : i + 1]
        label = label[i : i + 1]
        label_katakana = onehot_to_str(label)

        # 軸をN,H,W,Cに入れ替え
        data = data.transpose(0, 2, 3, 1)

        # ImageDataGeneratorのオブジェクト生成
        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.1,
            shear_range=0.2,
            rotation_range=15,
        )

        # 生成後枚数
        if label_katakana in ["a", "i", "u", "e", "o"]:
            num_image = 7 * n
        else:
            num_image = 4 * n

        # 生成
        g = datagen.flow(data)
        for i in range(num_image):
            batches = g.next()
            X.append(batches)
            Y.append(label)
    return np.array(X), np.array(Y)

if __name__ == '__main__':
    AUGUMENT = False
    train_data_path= "/home/students/workspace/SkillupAI/DeePL講座/DAY1_vr6_0_0/4_kadai/1_data/train_data.npy"
    train_label_path= "/home/students/workspace/SkillupAI/DeePL講座/DAY1_vr6_0_0/4_kadai/1_data/train_label.npy"
    X2 = np.load(train_data_path)
    Y2 = np.load(train_label_path)
    if AUGUMENT:
        X, Y = data_augumenter(train_data_path, train_label_path)
        X2 = np.squeeze(X, 1).copy()
        Y2 = np.squeeze(Y, 1).copy()
        print(X2.shape)
        X2 = X2.transpose(0, 3, 1, 2)
        print(X2.shape)
    # for i in range(num_image):
    #     img = X2[
    #         i,
    #         :,
    #         :,
    #         :,
    #     ]
    #     plt.imshow(img[0, :, :], cmap="gray")
    #     plt.show()


    # 正規化
    X2 = (X2 - X2.min()) / X2.max()
    X2 = X2.astype("float32")
    print("train_data.shape=", X2.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X2, Y2, test_size=0.3, random_state=1234, shuffle=True
    )
    print(X_train.shape, X_test.shape)


    X_train = X_train[:128]
    y_train = y_train[:128]


    # ## モデル作成


    epochs = 10
    batch_size = 32
    lr = 0.001
    early_stopping = 5
    count = 0
    callback = 5
    optimizer = Adam(lr)
    # 繰り返し回数
    xsize = X_train.shape[0]
    last_batch_size = xsize % batch_size
    assert last_batch_size == 0
    iter_num = np.ceil(xsize / batch_size).astype(np.int)



    model = CustomConvNet(
        input_dim=(1, 28, 28),
        conv_param={"filter_num": 30, "filter_size": 3, "pad": 0, "stride": 1},
        pool_param={"pool_size": 2, "pad": 0, "stride": 2},
        batch_size=batch_size,
        hidden_size=100,
        output_size=15,
        weight_init_std=0.01,
    )


    ## 学習
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    for epoch in range(epochs):
        # シャッフル
        idx = np.arange(xsize)
        np.random.shuffle(idx)

        for it in tqdm(range(iter_num)):
            """
            ランダムなミニバッチを順番に取り出す
            """
            mask = idx[batch_size*it : batch_size*(it+1)]

            # ミニバッチの生成
            x_ = X_train[mask]
            y_ = y_train[mask]

            # 勾配の計算
    #         if len(mask) == last_batch_size:
    #                 model.batch_size = last_batch_size
    #                 model.params['gamma1'] = np.ones((model.batch_size*model.conv_output_size**2, model.filter_num))
    #                 model.params['beta1'] = np.zeros((model.batch_size*model.conv_output_size**2, model.filter_num))
    #                 model.params['gamma2'] = np.ones((model.batch_size*model.conv2_output_size**2, model.filter_num))
    #                 model.params['beta2'] = np.zeros((model.batch_size*model.conv2_output_size**2, model.filter_num))
            grads = model.gradient(x_, y_)
            
            # パラメータの更新
            
            optimizer.update(model.params, grads)

        ## 学習経過の記録

        # 訓練データにおけるloss
        train_loss.append(model.loss(X_train,  y_train))

        # テストデータにおけるloss
        test_loss.append(model.loss(X_test, y_test))

        # 訓練データにて精度を確認
        train_accuracy.append(model.accuracy(X_train, y_train))

        # テストデータにて精度を算出
        test_accuracy.append(model.accuracy(X_test, y_test))
        
        # 最良な重み保存
        if epoch == 0:
            best_test_loss = test_loss[epoch]
        if best_test_loss >= test_loss[epoch]:
                best_model_params = model.params

        if epoch == 0 or (epoch + 1)  % callback == 0: 
            print("Epoch %d" % (epoch + 1), "train_loss", train_loss[epoch],"train_accuracy",train_accuracy[epoch], "test_loss", test_loss[epoch], "test accuracy", test_accuracy[epoch])
        
        #    早期終了(Early Stopping)
        if test_loss[epoch] >= test_loss[epoch-1]:
            count += 1
            if count >= early_stopping:
                print("Early Stopping !")
                break
        else:
            count=0


    # lossとaccuracyのグラフ化
    df_log = pd.DataFrame(
        {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        }
    )

    df_log.plot(style=["r-", "r--", "b-", "b--"])
    plt.ylim([0, 3])
    plt.ylabel("Accuracy or loss")
    plt.xlabel("epochs")
    plt.show()


    model.params = best_model_params
    accuracy = model.accuracy(X_test, y_test)
    loss = model.loss(X_test, y_test)
    print(accuracy, loss)


    from datetime import datetime as dt

    from pytz import timezone

    tdatetime = dt.now(timezone("Asia/Tokyo"))
    tstr = tdatetime.strftime("%Y%m%d%H%M")


    # ## 学習済みモデルの出力


    with open("katakana_model_weight_%s.pickle" % tstr, "wb") as f:
        pickle.dump(best_model_params, f)
