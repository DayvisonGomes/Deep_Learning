import numpy as np

def zeros(linhas, colunas):
    return np.zeros((linhas,colunas))

def ones(linhas, colunas):
    return np.ones((linhas,colunas))

def random_normal(linhas , colunas):
    return np.random.randn(linhas , colunas)

def random_uniform(rows , cols):
    return np.random.rand(rows , cols)

def glorot_normal(rows , cols):
    std_dev = np.sqrt(2.0 / (rows , cols))
    return std_dev * np.random.randn(rows , cols)

class Perceptron_():

    def __init__(self, dim_entrada, dim_saida, ini_pesos=random_normal, learning_rate=1e-3):
        self.learning_rate = learning_rate
        self.pesos = ini_pesos(dim_saida,dim_entrada)
        self.bias = np.random.uniform(-1,1)
        self.multi_class = False
        self.dim_saida = dim_saida
        
    def step(x):
        if x > 0:
            return 1
        else:
            return 0
    
    def fit(self, X_train, y_train, epochs=100, verbose=10):
        
        if self.dim_saida > 1:
            self.multi_class = True ## O código abaixo é para n neurônios na camada de saída, com mais de duas classificações

            for epoch in range(epochs + 1):
                cost = 0

                for x_n,y_n in zip(X_train, y_train):
                    y = np.dot(x_n, self.pesos.T)
                    y = np.argmax(y)
                    error = y_n - y
                    cost += error**2

                    self.__atualiza_pesos(x_n,y_n, y)            
            
                if epoch % verbose == 0:
                    print('epoch: {0:=4}/{1} loss_train: {2:.8f}'.format(epoch,epochs,0.5*(cost/(len(X_train)))))
        else:
            ## O código abaixo é para um neurônio só, ou seja, classificação binária
            for epoch in range(epochs + 1):
                cost = 0

                for x_n,y_n in zip(X_train, y_train):
                    y = np.dot(x_n, self.pesos.T) + self.bias
                    y = self.step(y)
                    error = y_n - y
                    cost += error**2

                    self.__atualiza_pesos(x_n, y_n,y)           
            
                if epoch % verbose == 0:
                    print('epoch: {0:=4}/{1} loss_train: {2:.8f}'.format(epoch,epochs,0.5*(cost/(len(X_train)))))
            
                   
    def predict(self, x):
        if self.multi_class:
            return np.argmax( np.dot(x, self.pesos.T) , axis=1 )

        return self.step( np.dot(x, self.pesos.T) + self.bias)

    def __atualiza_pesos(self, x_n, y_n, y):
        
        if self.multi_class:
            if not(y_n == y):
               self.pesos[int(y_n)] += self.learning_rate*x_n
               self.pesos[int(y)] -= self.learning_rate*x_n
    
        else:
            self.pesos = self.pesos + self.learning_rate*np.dot((y_n-y),x_n)
            self.bias = self.bias + self.learning_rate*(y_n-y)   
