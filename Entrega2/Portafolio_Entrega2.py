import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, recall_score

# Cargar datos
csv_path = "https://raw.githubusercontent.com/a01753010-ui/Portafolio1_Emilio_Torres/main/Entrega2/student_performance_dataset.csv"
df = pd.read_csv(csv_path)
X = df[['previous_grade', 'study_time_hours']]
y = df['final_exam_score']

# División 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
scaler = MinMaxScaler().fit(X_train)

# Red neuronal 2-3-1 con activación sigmoide (igual al modelo pasado)
nn = MLPRegressor(hidden_layer_sizes=(3,), activation='logistic', max_iter=10000, random_state=20)
nn.fit(scaler.transform(X_train), y_train)

# Predicción y evaluación
pred = nn.predict(scaler.transform(X_test))
print(f"MSE: {mean_squared_error(y_test, pred)}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred))}")

# Curva de convergencia del entrenamiento
plt.plot(nn.loss_curve_)
plt.xlabel('Iteración'); plt.ylabel('Pérdida (loss)')
plt.title('Convergencia del entrenamiento')
plt.show()

# Clasificación aprobado/reprobado (umbral 70)
y_true = (y_test >= 70).astype(int)
y_pred = (pred >= 70).astype(int)
print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
print(f"Recall:   {recall_score(y_true, y_pred)}")

# Matriz de confusión con umbral 70
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Reprobó', 'Aprobó'], yticklabels=['Reprobó', 'Aprobó'])
plt.xlabel('Predicción'); plt.ylabel('Real'); plt.title('Matriz de Confusión')
plt.show()

# Clasificación aprobado/reprobado (umbral 83.8)
y_true = (y_test >= 83.8).astype(int)
y_pred = (pred >= 83.8).astype(int)
print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
print(f"Recall:   {recall_score(y_true, y_pred)}")

# Matriz de confusión con umbral 83.8
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Reprobó', 'Aprobó'], yticklabels=['Reprobó', 'Aprobó'])
plt.xlabel('Predicción'); plt.ylabel('Real'); plt.title('Matriz de Confusión')
plt.show()

#FRONTERA DE DECISIÓN 2D
# Umbral por mediana (divide los datos en dos grupos de igual tamaño)
umbral = y.median()

# Malla que cubre todo el espacio de las 2 variables de entrada
x_min, x_max = X['previous_grade'].min(), X['previous_grade'].max()
y_min, y_max = X['study_time_hours'].min(), X['study_time_hours'].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

# Predecir en cada punto de la malla y convertir a aprobado/reprobado
Z = (nn.predict(scaler.transform(grid)) >= umbral).reshape(xx.shape)

# Frontera (regiones) + estudiantes reales del test encima
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')
plt.scatter(X_test['previous_grade'], X_test['study_time_hours'],
            c=(y_test >= umbral), cmap='RdYlGn', edgecolor='k', s=30)
plt.xlabel('previous_grade'); plt.ylabel('study_time_hours')
plt.title(f'Frontera de decisión (umbral = {umbral:.1f})')
plt.show()

#FRONTERA DE DECISIÓN 3D (con apoyo de IA)
umbral = y.median()

# Malla sobre las 2 variables de entrada
x_min, x_max = X['previous_grade'].min(), X['previous_grade'].max()
y_min, y_max = X['study_time_hours'].min(), X['study_time_hours'].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 60),
                     np.linspace(y_min, y_max, 60))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = nn.predict(scaler.transform(grid)).reshape(xx.shape)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

# Superficie de predicción del modelo
ax.plot_surface(xx, yy, Z, cmap='viridis', alpha=0.6)

# Plano horizontal en la altura del umbral
ax.plot_surface(xx, yy, np.full_like(Z, umbral), color='red', alpha=0.15)

# Estudiantes reales del conjunto de prueba
ax.scatter(X_test['previous_grade'], X_test['study_time_hours'], y_test,
           c=(y_test >= umbral), cmap='RdYlGn', edgecolor='k', s=25)

ax.set_xlabel('previous_grade')
ax.set_ylabel('study_time_hours')
ax.set_zlabel('final_exam_score')
ax.set_title(f'Superficie de predicción (plano rojo = umbral {umbral:.1f})')
ax.view_init(elev=20, azim=-60)
plt.show()