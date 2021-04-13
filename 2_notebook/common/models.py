# coding: utf-8
# import numpy as np
# import logging
from collections import OrderedDict

# import numpy as np
import cupy as np

# from common.functions import *
from common.activations import *
from common.layers import *
from common.loss import cross_entropy_error
from common.util import col2im, im2col


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict() # 順番付きdict形式. ただし、Python3.6以降は、普通のdictでもよい
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss() # 出力層
        
    def predict(self, x):
        """
        推論関数
        x : 入力
        """
        for layer in self.layers.values():
            # 入力されたxを更新していく = 順伝播計算
            x = layer.forward(x)
        
        return x
        
    def loss(self, x, t):
        """
        損失関数
        x:入力データ, t:教師データ
        """
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x, t):
        """
        識別精度
        """
        # 推論. 返り値は正規化されていない実数
        y = self.predict(x)
        #正規化されていない実数をもとに、最大値になるindexに変換する
        y = np.argmax(y, axis=1)
        
        if t.ndim != 1 : 
            """
            one-hotベクトルの場合、教師データをindexに変換する
            """
            t = np.argmax(t, axis=1)
        
        # 精度
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    def gradient(self, x, t):
        """
        全パラメータの勾配を計算
        """
        
        # 順伝播
        self.loss(x, t)

        # 逆伝播
        dout = 1 # クロスエントロピー誤差を用いる場合は使用されない
        dout = self.lastLayer.backward(dout=1) # 出力層
        
        ## doutを逆向きに伝える 
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # dW, dbをgradsにまとめる
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
    

    def numerical_gradient(self, x, t):
        """
        勾配確認用
        x:入力データ, t:教師データ        
        """
        
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
    
    
class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 pool_param={'pool_size':2, 'pad':0, 'stride':2},
                 hidden_size=100, output_size=10, weight_init_std=0.01, pretrained_weights=None):
        """
        input_size : tuple, 入力の配列形状(チャンネル数、画像の高さ、画像の幅)
        conv_param : dict, 畳み込みの条件
        pool_param : dict, プーリングの条件
        hidden_size : int, 隠れ層のノード数
        output_size : int, 出力層のノード数
        weight_init_std ： float, 重みWを初期化する際に用いる標準偏差
        """
                
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        
        pool_size = pool_param['pool_size']
        pool_pad = pool_param['pad']
        pool_stride = pool_param['stride']
        
        input_size = input_dim[1]
        conv_output_size = (input_size + 2*filter_pad - filter_size) // filter_stride + 1 # 畳み込み後のサイズ(H,W共通)
        pool_output_size = (conv_output_size + 2*pool_pad - pool_size) // pool_stride + 1 # プーリング後のサイズ(H,W共通)
        pool_output_pixel = filter_num * pool_output_size * pool_output_size # プーリング後のピクセル総数

        # 重みの初期化
        self.params = {}

        if pretrained_weights:
            self.params = pretrained_weights
        
        else:
            std = weight_init_std

            # W1は畳み込みフィルターの重み 
            # 配列形状=(フィルター枚数, チャンネル数, フィルター高さ, フィルター幅)
            self.params['W1'] =  std * np.random.randn(filter_num,input_dim[0],filter_size,filter_size)

            # b1は畳み込みフィルターのバイアス
            # 配列形状=(フィルター枚数)
            self.params['b1'] = np.zeros(filter_num) 

            # 全結合層の重みW
            # 配列形状=(前の層のノード数, 次の層のノード数)         
            self.params['W2'] = std * np.random.randn(pool_output_pixel,hidden_size)

            # 全結合層のバイアスb
            # 配列形状=(次の層のノード数)        
            self.params['b2'] = np.zeros(hidden_size)

            # 全結合層の重みW
            # 配列形状=(前の層のノード数, 次の層のノード数)        
            self.params['W3'] = std * np.random.randn(hidden_size,output_size)

            # 全結合層のバイアスb
            # 配列形状=(次の層のノード数)        
            self.params['b3'] = np.zeros(output_size)

        
        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad']) # W1が畳み込みフィルターの重み, b1が畳み込みフィルターのバイアスになる
        self.layers['ReLU1'] = ReLU()
        self.layers['Pool1'] = MaxPooling(pool_h=pool_size, pool_w=pool_size, stride=pool_stride, pad=pool_pad)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['ReLU2'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """
        損失関数
        x : 入力データ
        t : 教師データ
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def gradient(self, x, t):
        """勾配を求める（誤差逆伝播法）
        Parameters
        ----------
        x : 入力データ
        t : 教師データ
        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

class CustomConvNet_withBatchNorm:
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 pool_param={'pool_size':2, 'pad':0, 'stride':2},
                 batch_size = 32, hidden_size=100, output_size=10, weight_init_std=0.01, pretrained_weights=None, dropout_ration=0.5):
        """
        input_size : tuple, 入力の配列形状(チャンネル数、画像の高さ、画像の幅)
        conv_param : dict, 畳み込みの条件
        pool_param : dict, プーリングの条件
        hidden_size : int, 隠れ層のノード数
        output_size : int, 出力層のノード数
        weight_init_std ： float, 重みWを初期化する際に用いる標準偏差
        """
        
        self.filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        
        pool_size = pool_param['pool_size']
        pool_pad = pool_param['pad']
        pool_stride = pool_param['stride']
        
        input_size = input_dim[1]
        self.conv_output_size = (input_size + 2*filter_pad - filter_size) // filter_stride + 1 # 畳み込み後のサイズ(H,W共通)
        pool_output_size = (self.conv_output_size + 2*pool_pad - pool_size) // pool_stride + 1 # プーリング後のサイズ(H,W共通)
        pool_output_pixel = self.filter_num * pool_output_size * pool_output_size # プーリング後のピクセル総数
        
        self.conv2_output_size = (pool_output_size + 2*filter_pad - filter_size) // filter_stride + 1 # 畳み込み後のサイズ(H,W共通)
        pool2_output_size = (self.conv2_output_size + 2*pool_pad - pool_size) // pool_stride + 1 # プーリング後のサイズ(H,W共通)
        pool2_output_pixel = self.filter_num * pool2_output_size * pool2_output_size # プーリング後のピクセル総数
        # 重みの初期化
        self.params = {}
        
        # バッチサイズの初期化
        self.batch_size = batch_size
        
        if pretrained_weights:
            self.params = pretrained_weights
        
        else:
            std = weight_init_std


            # W1は畳み込みフィルターの重み 
            # 配列形状=(フィルター枚数, チャンネル数, フィルター高さ, フィルター幅)
            self.params['W1'] =  std * np.random.randn(self.filter_num,input_dim[0],filter_size,filter_size)

            # b1は畳み込みフィルターのバイアス
            # 配列形状=(フィルター枚数)
            self.params['b1'] = np.zeros(self.filter_num) 
            
            # バッチ正規化
            self.params['gamma1'] = np.ones((self.batch_size*self.conv_output_size**2, self.filter_num))
            self.params['beta1'] = np.zeros((self.batch_size*self.conv_output_size**2, self.filter_num))

            # W2は畳み込みフィルターの重み 
            # 配列形状=(フィルター枚数, チャンネル数, フィルター高さ, フィルター幅)
            self.params['W2'] =  np.random.randn(self.filter_num,self.filter_num,filter_size,filter_size)

            # b2は畳み込みフィルターのバイアス
            # 配列形状=(フィルター枚数)
            self.params['b2'] = np.zeros(self.filter_num) 
            
            # バッチ正規化
            self.params['gamma2'] = np.ones((self.batch_size*self.conv2_output_size**2, self.filter_num))
            self.params['beta2'] = np.zeros((self.batch_size*self.conv2_output_size**2, self.filter_num))

            # 全結合層の重みW
            # 配列形状=(前の層のノード数, 次の層のノード数)         
            self.params['W3'] = std * np.random.randn(pool2_output_pixel,hidden_size)

            # 全結合層のバイアスb
            # 配列形状=(次の層のノード数)        
            self.params['b3'] = np.zeros(hidden_size)

            # 全結合層の重みW
            # 配列形状=(前の層のノード数, 次の層のノード数)         
            self.params['W4'] = std * np.random.randn(hidden_size,output_size)

            # 全結合層のバイアスb
            # 配列形状=(次の層のノード数)        
            self.params['b4'] = np.zeros(output_size)
            
            

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad']) # W1が畳み込みフィルターの重み, b1が畳み込みフィルターのバイアスになる
    
        self.layers['ReLU1'] = ReLU()
        self.layers['BatchNorm1'] = BatchNormalization(self.params['gamma1'], self.params['beta1'])

        self.layers['Pool1'] = MaxPooling(pool_h=pool_size, pool_w=pool_size, stride=pool_stride, pad=pool_pad)
        
        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'],
                                           conv_param['stride'], conv_param['pad']) # W2が畳み込みフィルターの重み, b2が畳み込みフィルターのバイアスになる
        
        self.layers['ReLU2'] = ReLU()
        self.layers['BatchNorm2'] = BatchNormalization(self.params['gamma2'], self.params['beta2'])

        self.layers['Pool2'] = MaxPooling(pool_h=pool_size, pool_w=pool_size, stride=pool_stride, pad=pool_pad)
        
        self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['ReLU3'] = ReLU()
        
#         self.layers['Dropout1'] = Dropout(dropout_ration)
        
        self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['ReLU4'] = ReLU()


        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flg=False):
        """
        損失関数
        x : 入力データ
        t : 教師データ
        """
        y = self.predict(x, train_flg)
        return self.last_layer.forward(y, t)
    
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def gradient(self, x, t):
        """勾配を求める（誤差逆伝播法）
        Parameters
        ----------
        x : 入力データ
        t : 教師データ
        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W4'], grads['b4'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        grads['gamma1'] = self.layers['BatchNorm1'].dgamma
        grads['beta1'] = self.layers['BatchNorm1'].dbeta
        grads['gamma2'] = self.layers['BatchNorm2'].dgamma
        grads['beta2'] = self.layers['BatchNorm2'].dbeta

        return grads


# class Resnet:

#     def __init__(self, input_dim=(1, 28, 28), 
#                  conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
#                  pool_param={'pool_size':2, 'pad':0, 'stride':2},
#                  hidden_size=100, output_size=10, weight_init_std=0.01, pretrained_weights=None) -> None:

#         """
#         input_size : tuple, 入力の配列形状(チャンネル数、画像の高さ、画像の幅)
#         conv_param : dict, 畳み込みの条件
#         pool_param : dict, プーリングの条件
#         hidden_size : int, 隠れ層のノード数
#         output_size : int, 出力層のノード数
#         weight_init_std ： float, 重みWを初期化する際に用いる標準偏差
#         """
        