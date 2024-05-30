import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar datos de ejemplo
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
data = pd.read_csv(url, header=None)

# Asignar nombres de columnas
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Ver los primeros registros
print(data.head())

# Manejo de valores nulos (en este caso no hay, pero es un ejemplo)
data = data.dropna()

# Normalizacion y escalado
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.iloc[:, :-1])

# Division de los datos
X_train, X_test, y_train, y_test = train_test_split(scaled_data, data['class'], test_size=0.2, random_state=42)

# Ver las dimensiones de los conjuntos de los datos
print("Conjunto de entrenamiento: ", X_train.shape)
print("Conjunto de prueba: ", X_test.shape)