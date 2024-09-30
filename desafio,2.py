import tensorflow as tf    
from sklearn.datasets import load_diabetes  
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = load_diabetes()


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33, random_state=42)

model = tf.keras.models.Sequential([

    tf.keras.layers.Dense(500, activation=tf.nn.relu), 
       tf.keras.layers.Dense(250, activation=tf.nn.relu),
          tf.keras.layers.Dense(125, activation=tf.nn.relu),
             tf.keras.layers.Dense(50, activation=tf.nn.relu), 
    tf.keras.layers.Dense(1, activation='linear')  
])

model.compile(optimizer='adam',
              loss='mean_squared_error', 
              metrics=['mae']) 


model.fit(x_train, y_train, epochs=100)


model.evaluate(x_test, y_test)
previsoes = model.predict(x_test)


mae = mean_absolute_error(y_test, previsoes)
mse = mean_squared_error(y_test, previsoes)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, previsoes)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R²: {r2}')


plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Valores Reais', alpha=0.6)
plt.scatter(range(len(previsoes)), previsoes, color='red', label='Previsões', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfeito') 

plt.title('Valores Reais vs Previsões')
plt.xlabel('valores reais')
plt.ylabel('previsoes')
plt.legend()
plt.show()
