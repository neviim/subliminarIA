# -*- coding: UTF-8 -*-
#
import tensorflow as tf
import keras

import matplotlib.pyplot as plt
import numpy as np

from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras.preprocessing import image
from keras import backend as K
from PIL import Image

# ---

__ghost__ = "Neviim Jads"

# download, Modelos de classificação de imagem treinados para Keras.
iv3 = InceptionV3()

# le a imagem em uma matrix de valor, usando tamanho de (299 x 299)
X = image.img_to_array(image.load_img("../image/cachorro.jpg", target_size=(299, 299))) 

# Alterando os valores da matrix de um ranger de 0-255 para um de -1 a 1
X /= 255
X -= 0.5
X *= 2

# Dimensão da rede: (299,299,3)
# print(X.shape)
X = X.reshape([1, X.shape[0], X.shape[1], X.shape[2]])
# Acrecentar uma nova colona dimencional 1: (1,299,299,3)
# print(X.shape)

# aplicar o calculo de predição Linear e controle de variancia minima
# (http://users.isr.ist.utl.pt/~alex/micd0506/micd6a.pdf)
Y = iv3.predict(X)

# apos processamento retorna a classificação processada.
resultado = decode_predictions(Y)

# imprime o resultado formatado.
for id, nome, probabilidade in resultado[0]:
    print(f'{id:10} -  {nome:25} >> {probabilidade:10f}')

# --- reverte.
X /= 2
X += 0.5
X *= 255

# salva imagem no formato 299x299, a qual o tensorflow aceita.
image = Image.fromarray(X[0].astype(np.uint8)) 
image.save("../image/cachorro_299x299.png")
