import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

# Cargar datos
csv_path = "https://raw.githubusercontent.com/a01753010-ui/Portafolio1_Emilio_Torres/main/Entrega2/student_performance_dataset.csv"
df = pd.read_csv(csv_path)
X = df[['previous_grade', 'study_time_hours', 'sleep_hours', 'attendance_percent']]
y = df['final_exam_score']

# División 70/30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
scaler = MinMaxScaler().fit(X_train)

# Red neuronal 2-5-1 con activación relu
nn = MLPRegressor(hidden_layer_sizes=(5,), activation='relu', max_iter=10000, random_state=20)
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
print(f"Precision: {precision_score(y_true, y_pred)}")
print(f"F1 Score: {f1_score(y_true, y_pred)}")

# Matriz de confusión con umbral 83.8
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Reprobó', 'Aprobó'], yticklabels=['Reprobó', 'Aprobó'])
plt.xlabel('Predicción'); plt.ylabel('Real'); plt.title('Matriz de Confusión')
plt.show()