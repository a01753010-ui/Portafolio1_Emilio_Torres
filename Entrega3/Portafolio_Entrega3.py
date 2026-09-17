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
# print(f"MSE: {mean_squared_error(y_test, pred)}")
# print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred))}")


# PARTE 3

# DATOS TEST
csv_path_test = "https://raw.githubusercontent.com/a01753010-ui/Portafolio1_Emilio_Torres/main/Entrega3/student_performance_updated_1000.csv"

# Cargar el dataset nuevo
df_test = pd.read_csv(csv_path_test)

# Eliminar filas donde falta la calificación real
df_test = df_test.dropna(subset=['FinalGrade'])

# Rellenar faltantes con la media del entrenamiento original
df_test['PreviousGrade'] = df_test['PreviousGrade'].fillna(X_train['previous_grade'].mean())
df_test['Study Hours']   = df_test['Study Hours'].fillna(X_train['study_time_hours'].mean())

# Seleccionar variables y renombrarlas para que coincidan con los nombres que el scaler y el modelo aprendieron
X_nuevo = df_test[['PreviousGrade', 'Study Hours']].rename(columns={
    'PreviousGrade': 'previous_grade',
    'Study Hours':   'study_time_hours'
})
y_nuevo = df_test['FinalGrade']

# 3. Escalar las entradas con el MISMO scaler ya ajustado
X_nuevo_scaled = scaler.transform(X_nuevo)

# 4. Predecir con el modelo ya entrenado
pred_nuevo = nn.predict(X_nuevo_scaled)

# 5. Evaluar el desempeño sobre datos nunca vistos
print(f"MSE (datos nuevos):  {mean_squared_error(y_nuevo, pred_nuevo)}")
print(f"RMSE (datos nuevos): {np.sqrt(mean_squared_error(y_nuevo, pred_nuevo))}")


# Clasificación aprobado/reprobado (umbral 80 utilizando mediana)
y_nuevo_bin = (y_nuevo >= 80).astype(int)
pred_nuevo_bin = (pred_nuevo >= 80).astype(int)

print(f"Accuracy: {accuracy_score(y_nuevo_bin, pred_nuevo_bin)}")
print(f"Recall:   {recall_score(y_nuevo_bin, pred_nuevo_bin)}")
print(f"Precision: {precision_score(y_nuevo_bin, pred_nuevo_bin)}")
print(f"F1 Score: {f1_score(y_nuevo_bin, pred_nuevo_bin)}")

# Matriz de confusión con umbral 80
plt.figure()
sns.heatmap(confusion_matrix(y_nuevo_bin, pred_nuevo_bin), annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Reprobó', 'Aprobó'], yticklabels=['Reprobó', 'Aprobó'])
plt.xlabel('Predicción'); plt.ylabel('Real'); plt.title('Matriz de Confusión')
plt.show()