# -*- coding: UTF-8 -*-

""" Subliminar IA

     Returns:
        Tem como entrada uma imagem qualquer, ela é processada e retorna uma analize 
        de algumas probabilidades com um grau de % de reconhecimento da mesma.

        Posteriormente é realizado um algoritimo de manipulação de pixel, criando uma 
        alteração na forma com a qual o neironio passe a interpretar esta imagem como
        sendo um limão com margem de mais de 90% de probabilidade de certesa.

        Isso mostra que é possivel enganar uma rede neural assim como nosso celebro 
        pode ser enganado atraves de imagens no que chamanos de imagens subliminar.
"""

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

class ReconheceImagem(object):
    """ [Classe que identifica e manupula imagem por rede neural]
    
        Arguments:
            object {[type]} -- [description]
        
        Returns:
            [type] -- [description]
    """

    def __init__(self):
        self.iv3 = InceptionV3()
        self.hackeada = []
        self.adv = []
        self.X   = []

    def identifica(self, nomeImagem):
        """ [Identificação de imagem por rede neural]
        
            Arguments:
                nomeImagem {[string]} -- [nome da imagem a ser identificada pela rede neural]
            
            Returns:
                [Lista] -- [Contendo as 5 identificaçoes encontrada da possibilidade 
                            de reconhecimento da imagem recebida.]
        """

        # Rede neural identificando imagem.
        ar_X = image.img_to_array(image.load_img(nomeImagem, target_size=(299, 299))) 

        # Alterando os valores da matrix de um ranger de 0-255 para -1 a 1.
        ar_X /= 255
        ar_X -= 0.5
        ar_X *= 2

        # aplicando reshape.
        ar_X = ar_X.reshape([1, ar_X.shape[0], ar_X.shape[1], ar_X.shape[2]])
        ar_Y = self.iv3.predict(ar_X)

        # apos processamento retorna a classificação processada.
        resultado = decode_predictions(ar_Y)
        self.X = np.copy(ar_X)

        return resultado

    def manipula(self):
        """ [ Metodo para manipular os pixel de uma imagem ]
        
            Returns:
                [Array] -- [array modificado.]
        """

        # Criando uma forma de manipular esta rede
        inp_layer = self.iv3.layers[0].input
        out_layer = self.iv3.layers[-1].output

        # 951 - como a classe_limão esta identificada.
        target_class = 951
        loss = out_layer[0, target_class]
        grad = K.gradients(loss, inp_layer)[0]
        optimize_gradient = K.function([inp_layer, K.learning_phase()], [grad, loss])

        self.adv = np.copy(self.X)

        # valor de pertubação que não pode ser ultrapaçado
        # esta referencia é para deixar a imagem gerada imperceptivel 
        # a nos humanos, na manipulação do pixel não pode ultrapassar 
        # estes valores:
        pert = 0.01
        max_pert = self.X + 0.01
        min_pert = self.X - 0.01

        # loop de processamento.
        cost = 0.0
        while cost < 0.95:
        
            gr, cost = optimize_gradient([self.adv, 0]) # ,(0) indica modo teste.            
            self.adv += gr
            
            # os valores de saturação não pode ser maiores que estas variaveis
            self.adv = np.clip(self.adv, min_pert, max_pert)
            self.adv = np.clip(self.adv, -1, 1)
            
            print("Probabilidade de ser um limão: ", cost)

        # reinverte este passo realizado em (self.identifica) 0-255 para -1 a 1.
        self.adv /= 2
        self.adv += 0.5
        self.adv *= 255

        # tira uma copia das variaveis.
        self.hackeada = np.copy(self.adv)

        # Salva imagem hackeada.
        novoImagem = Image.fromarray(self.hackeada[0].astype(np.uint8))
        novoImagem.save("../image/imagemHackeada.png")

        return self.hackeada


# Iniciando...

if __name__ == "__main__":
    """ [summary]
    """
    # instancia classe de reconhecimento.
    imagemHacker = ReconheceImagem()
    
    # resultara que a imagem é 70% provavel ser de um Labrador Retriever
    probImagemOriginal = imagemHacker.identifica("../image/cachorro.jpg")
    imagemAdulterada   = imagemHacker.manipula()
    probImagemKackeada = imagemHacker.identifica("../image/imagemHackeada.png")

    print()
    print("Resultado da imagem original:")
    print()

    # mostra resultado obtido da imagem original analizada.
    for id, nome, prob in probImagemOriginal[0]:
        print(f'{id:10} -  {nome:25} >> {prob:10f}')
    print()
    # ----

    print()
    print("Resultado da imagem alterada:")
    print()

    # mostra resultado obtido da imagem hackeada analizada.
    for id, nome, prob in probImagemKackeada[0]:
        print(f'{id:10} -  {nome:25} >> {prob:10f}')
    print()
    # ----





