import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms

# Crear un conjunto de datos ficticio relacionado con el fútbol
np.random.seed(42)
edad = np.random.randint(18, 60, 100)  # Edades entre 18 y 60 años
ubicacion = np.random.randint(0, 2, 100)  # 0 para localidad A, 1 para localidad B
equipos = np.random.choice(['Equipo1', 'Equipo2', 'Equipo3'], 100)

data = np.column_stack((edad, ubicacion, equipos))
X = data[:, :-1]  # Características (edad y ubicación)
y = data[:, -1]   # Variable objetivo (equipo favorito)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir la función de aptitud (fitness)
def evaluate_model(params):
    n_estimators, max_depth = params
    model = RandomForestClassifier(n_estimators=int(n_estimators), max_depth=int(max_depth), random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return -accuracy  # Usamos el signo negativo porque queremos maximizar la precisión

# Crear un tipo de aptitud maximizado
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Crear un individuo con dos hiperparámetros (n_estimators y max_depth)
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 10, 100)  # Rango para n_estimators
toolbox.register("attr_int", random.randint, 2, 20)    # Rango para max_depth
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Función para evaluar la aptitud
def evaluate(individual):
    return evaluate_model(individual),

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[10, 2], up=[100, 20], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Crear una población inicial
population = toolbox.population(n=10)

# Ejecutar el algoritmo genético
algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=50, verbose=True)

# Obtener el mejor individuo
best_individual = tools.selBest(population, k=1)[0]
best_n_estimators, best_max_depth = best_individual

# Crear y entrenar el modelo optimizado
optimized_model = RandomForestClassifier(n_estimators=int(best_n_estimators), max_depth=int(best_max_depth), random_state=42)
optimized_model.fit(X_train, y_train)
y_pred_optimized = optimized_model.predict(X_test)
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)

# Comparar el rendimiento del modelo antes y después de la optimización de parámetros
print("Precisión antes de la optimización:", evaluate_model([100, 10]))
print("Precisión después de la optimización:", accuracy_optimized)
