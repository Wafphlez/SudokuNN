#!/usr/bin/env python
# coding: utf-8

# # Импорт

# In[1]:


import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from keras import Model, Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.utils import to_categorical


# # Методы:

# ### Загрузка данных игр в судоку из таблицы

# In[2]:


def load_data(nb_train=50000, nb_test=320, full=False):

    if full:
        sudokus = pd.read_csv('sudoku.csv').values
    else:
        sudokus = next(
            pd.read_csv('sudoku.csv', chunksize=(nb_train + nb_test))
        ).values
        
    quizzes, solutions = sudokus.T
    flatX = np.array([np.reshape([int(d) for d in flatten_grid], (9, 9))
                      for flatten_grid in quizzes])
    flaty = np.array([np.reshape([int(d) for d in flatten_grid], (9, 9))
                      for flatten_grid in solutions])
    
    return (flatX[:nb_train], flaty[:nb_train]), (flatX[nb_train:], flaty[nb_train:])


# ### Рассчет точности решений

# In[3]:


def diff(grids_true, grids_pred):
    return (grids_true != grids_pred).sum((1, 2))


# ### Удаление случайных ячеек в решенных играх

# In[4]:


def delete_digits(X, n_delet=1):
    grids = X.argmax(3)
    for grid in grids:
        grid.flat[np.random.randint(0, 81, n_delet)] = 0
        
    return to_categorical(grids)


# ### Поиск решения

# In[5]:


def batch_smart_solve(grids, solver):
    grids = grids.copy()
    for _ in range((grids == 0).sum((1, 2)).max()):
        preds = np.array(solver.predict(to_categorical(grids)))
        probs = preds.max(2).T
        values = preds.argmax(2).T + 1
        zeros = (grids == 0).reshape((grids.shape[0], 81))

        for grid, prob, value, zero in zip(grids, probs, values, zeros):
            if any(zero):
                where = np.where(zero)[0]
                confidence_position = where[prob[zero].argmax()]
                confidence_value = value[confidence_position]
                grid.flat[confidence_position] = confidence_value
    return grids


# # Подготовка данных к обучению

# In[17]:


load_data(nb_train=50000, nb_test=320, full=False)


# In[18]:


input_shape = (9, 9, 10)
(_, ytrain), (Xtest, ytest) = load_data()

Xtrain = to_categorical(ytrain).astype('float32')
Xtest = to_categorical(Xtest).astype('float32')

ytrain = to_categorical(ytrain-1).astype('float32')
ytest = to_categorical(ytest-1).astype('float32')


# # Определить модель keras

# In[19]:


model = Sequential()
model.add(Dense(64, activation='relu', input_shape=input_shape))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Flatten())

grid = Input(shape=input_shape)
features = model(grid)

digit_placeholders = [
    Dense(9, activation='softmax')(features)
    for i in range(81)
]

solver = Model(grid, digit_placeholders)
solver.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# # Итерационное обучение модели на тестовых данных

# In[20]:


history = solver.fit(
    delete_digits(Xtrain, 0),
    [ytrain[:, i, j, :] for i in range(9) for j in range(9)],
    batch_size=64,
    epochs=1,
    verbose=1,
)


# ## Решение судоку со случайно удаленными значениями

# In[21]:


early_stop = EarlyStopping(patience=2, verbose=1)

i = 1
for nb_epochs, nb_delete in zip(
        #[1, 2, 3, 4, 6, 8, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        [1, 1, 2, 3, 5, 5, 5],
        [1, 2, 5, 10, 20, 40, 55]
        #[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #[1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 55]
):
    print('Pass n° {} ...'.format(i))
    i += 1
    
    history = solver.fit(
        delete_digits(Xtrain, nb_delete),
        [ytrain[:, i, j, :] for i in range(9) for j in range(9)],
        validation_data=(
            delete_digits(Xtrain, nb_delete),
            [ytrain[:, i, j, :] for i in range(9) for j in range(9)]),
        batch_size=64,
        epochs=nb_epochs,
        verbose=1,
        callbacks=[early_stop]
    )


# # Результаты

# In[ ]:


quizzes = Xtest.argmax(3)
true_grids = ytest.argmax(3) + 1
smart_guesses = batch_smart_solve(quizzes, solver)

deltas = diff(true_grids, smart_guesses)
accuracy = (deltas == 0).mean()


# In[ ]:


print(
"""
Решено:\t {}
Корректных:\t {}
Точность:\t {}
""".format(
deltas.shape[0], (deltas==0).sum(), accuracy
)
)


# In[ ]:


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[ ]:





# In[ ]:




