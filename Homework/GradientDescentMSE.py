'''
Tеперь вам необходимо реализовать класс для оптимизации коэффициентов линейной регрессии МНК.
Подразумевается, что на вход алгоритм будет принимать следующие параметры:\
- 2 pandas датафрейма **samples** и **targets**, содержащих матрицу объектов и ветор ответов соответственно
- значение **learning rate**, который корректирует длину вектора-градиента (чтобы он не взорвался)
- значение **threshold**'а для критерия останова (когда мы считаем, что мы сошлись к оптимуму?)
- параметр **copy**, который позволяет либо делать изменения in-place в датафрейме, подающимся в класс, 
если изменения матрицы объектов в принципе при обучении имеются. 
Или же копировать объект при инициализации класса и возвращать новый объект, если требуется.
Он будет состоять из следующих важных компонент-методов:
- **add_constant_feature**: добавляет колонку с названием *constant* из единичек к переданному датафрейму **samples**. 
Это позволяет оценить свободный коэффициент $\beta_0$.
- **calculate_mse_loss**: вычисляет при текущих весах **self.beta** значение среднеквадритической ошибки.
- **calculate_gradient**: вычисляет при текущих весах вектор-градиент по функционалу.
- **iteration**: производит итерацию градиентного спуска, то есть обновляет веса модели, в соответствии с 
установленным **learning_rate = $\eta$**: $\beta^{(n+1)} = \beta^{(n)} - \eta \cdot \nabla Q(\beta^{(n)})$
- **learn**: производит итерации обучения до того момента, пока не сработает критерий останова обучения. В этот раз критерием останова будет следующее событие: во время крайней итерации изменение в функционале качества модели составило значение меньшее, чем **self.threshold**. Иными словами, $|Q(\beta^{(n+1)}) - Q(\beta^{(n+1)})| < threshold$.
P.S. установите в **__init__** аттрибут экземпляра с названием **iteration_loss_dict**, который будет устроен следующим 
образом: на каждой итерации мы будем добавлять в словарь пару ключ-значение, где ключем будет номер итерации $n$, 
а значением - среднеквадратическая ошибка в точке $\beta^{(n)}$. Это пригодится нам в будущем для визуализации.
'''
import pandas as pd
import numpy as np

class GradientDescentMse:

    def __init__(self, samples: pd.DataFrame, targets: pd.DataFrame,
                 learning_rate: float = 1e-3, threshold = 1e-6, copy: bool = True):
        self.iterations_loss_dict = {}
        if copy:
            self.samples = samples.copy()
        else:
            self.samples = samples
        self.targets = targets
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.beta = np.ones(self.samples.shape[1])
        self.iteration_num = 0

    def add_constant_feature(self, samples):
        self.samples['constant'] = 1
        self.beta = np.append(self.beta, 1)
        return self.samples, self.beta

    def calculate_mse_loss(self, samples, targets):
        self.mse_loss = np.mean((targets - samples.dot(self.beta))**2)
        return self.mse_loss
    
    def calculate_gradient(self, samples, targets):
        #self.gradient = -2 * self.samples.T.dot(self.targets - self.samples.dot(self.beta))
       
        shift = np.dot(self.samples, self.beta)  - self.targets.values
        self.gradient = 2 * np.dot(shift, self.samples) / self.samples.shape[0]
        return self.gradient

    def iteration(self, samples, targets):
        for i in range(1000):
            self.beta -= self.learning_rate * self.calculate_gradient(samples, targets)
            self.iterations_loss_dict[i] = self.calculate_mse_loss(samples, targets)

        return self.beta, self.iterations_loss_dict

    def learn(self, samples, targets):
        self.samples = samples
        self.targets = targets
        self.beta = np.zeros(samples.shape[1])
        self.iteration(samples, targets)
        return self.beta, self.iterations
