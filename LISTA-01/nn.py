import numpy as np
from Perceptron_ import random_normal,ones,zeros

def linear(x, derivada=False):
    return np.ones_like(x) if derivada else x

def sigmoid(x, derivada=False):
    if derivada:
        y = sigmoid(x)
        return y*(1 - y)
    return 1.0/(1.0 + np.exp(-x))

def tanh(x, derivada=False):
    if derivada:
        y = tanh(x)
        return 1 - y**2
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def relu(x, derivada=False):
    if derivada:
        return np.where(x <= 0, 0, 1)
    return np.maximum(0, x)

def softmax(x, y_oh=None, derivada=False):
    if derivada:
        y_pred = softmax(x)
        k = np.nonzero(y_pred * y_oh)
        pk = y_pred[k]
        y_pred[k] = pk * (1.0 - pk)
        return y_pred
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)

def mae(y , y_pred , derivada=False):
    if derivada:
        return np.where( y_pred > y , 1 , -1) / y.shape[0]
    return np.mean(np.abs(y - y_pred))

def mse(y , y_pred , derivada=False):
    if derivada:
        return (-(y - y_pred) )/ y.shape[0]

    return 0.5 * np.mean( (y - y_pred)**2 ) 

def cross_entropy(y , y_pred , derivada=False):
    if derivada:
        return  -  (y - y_pred )  / (y.shape[0] * y_pred * (1 - y_pred) )

    return -np.mean( y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred) )

def neg_log_likelihood(y_oh, y_pred, derivada=False):
    k = np.nonzero(y_pred * y_oh)
    pk = y_pred[k]
    if derivada:
        y_pred[k] = (-1.0 / pk)
        return y_pred
    return np.mean(-np.log(pk))

def softmax_neg_log_likelihood(y_oh, y_pred, derivada=False):
    y_softmax = softmax(y_pred)
    if derivada:
        return -(y_oh - y_softmax) / y_oh.shape[0]    
    return neg_log_likelihood(y_oh, y_softmax)


class Layer():

    def __init__(self, dim_entrada, dim_saida, ini_pesos=random_normal, ini_bias=ones, activation=linear) :
        self.entrada = None
        self.pesos = ini_pesos(dim_saida, dim_entrada)
        self.bias = ini_bias(1, dim_saida)
        self.f_ativacao = activation

        self._acti_entrada, self._acti_saida = None, None
        self._dpesos, self._dbias = None, None

class NeuralNetwork():

    def __init__(self, func_cost=mse, learning_rate=1e-3):
        self.layers = []
        self.func_cost = func_cost
        self.learning_rate = learning_rate

    def fit(self, X_train, y_train, epochs=100, verbose=10):

        for epoch in range(epochs + 1):
            y_pred = self.__feedforward(X_train)
            self.__backprop(y_train,y_pred)

            if epoch % verbose == 0:
                train_loss = self.func_cost(y_train,self.predict(X_train))
                print('epoch: {0:=4}/{1} loss_train: {2:.8f}'.format(epoch,epochs,train_loss))
    
    def predict(self, x):
        return self.__feedforward(x)
    
    def __feedforward(self, x):
        self.layers[0].entrada = x

        for atual, proxima in zip(self.layers, self.layers[1:] + [Layer(0,0)]):
            y = np.dot( atual.entrada, atual.pesos.T ) + atual.bias
            atual._acti_entrada = y
            atual._acti_saida = atual.f_ativacao(y)
            proxima.entrada = atual._acti_saida

        return self.layers[-1]._acti_saida

    def __backprop(self, y, y_pred):
        delta = self.func_cost(y, y_pred, derivada=True)  ## Derivada da função de custo

        for layer in reversed(self.layers):
            grandiente_local = layer.f_ativacao(layer._acti_entrada, derivada=True) * delta #sinal de erro na saída vezes derivada da função de ativação
            delta = np.dot(grandiente_local, layer.pesos) ## soma ponderada dos gradientes locais da camada seguinte
            layer._dpesos = np.dot(grandiente_local.T, layer.entrada) ## gradiente local vezes o sinal de entrada
            layer._dbias = 1*grandiente_local.sum(axis=0, keepdims=True) 

        for layer in reversed(self.layers):
            layer.pesos = layer.pesos - self.learning_rate * layer._dpesos
            layer.bias = layer.bias - self.learning_rate * layer._dbias

    