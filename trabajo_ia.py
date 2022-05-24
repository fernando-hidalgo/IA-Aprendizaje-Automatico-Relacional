# %%
from numpy.lib.function_base import copy
import pandas as pd
import networkx as nx
from funciones import n_b,r_lineal_multiple,r_logistica,arboles_decision,knn
from sklearn import tree


#Leemos los datos y le indicamos que no tenemos cabecera, quitamos la primero fila y columna que son  indices
matrix = pd.read_csv("CSV/LONDON_GANG.csv",header=None)
matrix = matrix.iloc[1: , 1:]

#Comprobamos que esta correctamente guardado
print(matrix.head())


atributos = pd.read_csv("CSV/LONDON_GANG_ATTR.csv",header=0)
atributos = atributos.iloc[:, 1:]

print(atributos.head())

#Creamos un grafo con la matriz de adyacencia y comprobamos que se ha creado correctamente, en nuestro caso lo dibujamos numerado
grafo = nx.Graph(matrix.values)
#nx.draw_networkx(grafo, node_size=130)


#Vamos a obtener los atributos relacionales, vienen como un diccionario asi que cogemos los valores y los ponemos en forma de lista
centralidad_grado = list(nx.degree_centrality(grafo).values())
centralidad_intermedia = list(nx.betweenness_centrality(grafo).values())
centralidad_katz = list(nx.katz_centrality_numpy(grafo, alpha=0.1, beta=1.0).values())

print("CENTRALIDAD DE GRADO\n")
print(centralidad_grado)
print("\n====================================================================================================================================================\n")
print("CENTRALIDAD DE INTERMEDIACIÓN\n")
print(centralidad_intermedia)
print("\n====================================================================================================================================================\n")
print("CENTRALIDAD DE KATZ\n")
print(centralidad_katz)
print("\n====================================================================================================================================================\n")

#Los grados vienen datos en tuplas con (indice,grado) de cada nodo, en nuestro caso solo  remos el grado
grados = [j for i,j in grafo.degree]
print(grados)
print("\n====================================================================================================================================================\n")

#Al igual que los anteriores metodos, devuelve un diccionario y pasamos los valores a una lista
clusters = list(nx.clustering(grafo).values())
print(clusters)
print("\n====================================================================================================================================================\n")

atributos_relacionales = {"Centralidad_grado":centralidad_grado, "Centralidad_intermedia":centralidad_intermedia, 
                          "Centralidad_katz":centralidad_katz, "Grados":grados, "Clusters":clusters}

#Creamos los dataframes con los nuevos datos juntos y separados
atributos_total = atributos.copy()
for k,v in atributos_relacionales.items():
    atributos_total[k] = v
    
datos = []    
for k,v in atributos_relacionales.items():
    atributos_aux = atributos.copy()
    atributos_aux[k] = v
    datos.append(atributos_aux)

#Datos a usar
t_size = 0.25
at_objetivo = "Prison"
objetivo = atributos[[at_objetivo]].values.ravel()
atributos.drop(axis=1, columns = [at_objetivo], inplace=True)
atributos_total.drop(axis=1, columns = [at_objetivo], inplace=True)

for i in range(0,len(datos)):
    datos[i].drop(axis=1, columns = [at_objetivo], inplace=True)

print("Se ha usado un porcetaje de entrenamiento de ", (1.0-t_size) * 100,"%\n")

#Naive Bayes
print("\nNAIVE BAYES\n")

print(f"Acierto con atributos originales para {at_objetivo} : {n_b(atributos, objetivo, t_size)}\n")
print(f"Acierto con todos los atributos para {at_objetivo} : {n_b(atributos_total, objetivo, t_size)}\n")
print(f"Acierto con los atributos relacionales solamente, para {at_objetivo}: ",n_b(atributos_total[["Centralidad_grado", "Centralidad_intermedia", "Centralidad_katz", "Grados", "Clusters"]],objetivo, t_size),"\n")

for i in range(0,len(datos)):
    dat_usado = datos[i]
    at_nuevo = datos[i].columns[7]
    print(f"Acierto con atributos originales  mas",at_nuevo,"para",at_objetivo,":",n_b(dat_usado, objetivo, t_size),"\n")


#Regresion Lineal Multiple
print("\nRegresión Lineal\n")

print(f"Acierto con atributos originales para {at_objetivo} : {r_lineal_multiple(atributos, objetivo, t_size)}\n")
print(f"Acierto con todos los atributos para {at_objetivo} : {r_lineal_multiple(atributos_total, objetivo, t_size)}\n")
print(f"Acierto con los atributos relacionales solamente, para {at_objetivo}: ",r_lineal_multiple(atributos_total[["Centralidad_grado", "Centralidad_intermedia", "Centralidad_katz", "Grados", "Clusters"]],objetivo, t_size),"\n")

for i in range(0,len(datos)):
    dat_usado = datos[i]
    at_nuevo = datos[i].columns[7]
    print(f"Acierto con atributos originales  mas",at_nuevo,"para",at_objetivo,":",r_lineal_multiple(dat_usado, objetivo, t_size),"\n")

#Regresión Logística
print("\nRegresión Logistica Test-Split\n")


print(f"Acierto con atributos originales para {at_objetivo} : {r_logistica(atributos, objetivo, t_size, 0, False)}\n")
print(f"Acierto con todos los atributos para {at_objetivo} : {r_logistica(atributos, objetivo, t_size, 0, False)}\n")
print(f"Acierto con los atributos relacionales solamente, para {at_objetivo}: ",r_logistica(atributos_total[["Centralidad_grado", "Centralidad_intermedia", "Centralidad_katz", "Grados", "Clusters"]],objetivo, t_size, 0, False),"\n")

for i in range(0,len(datos)):
    dat_usado = datos[i]
    at_nuevo = datos[i].columns[7]
    print(f"Acierto con atributos originales  mas",at_nuevo,"para",at_objetivo,":",r_logistica(dat_usado, objetivo, t_size, 0, False),"\n")


print("\nRegresión Logistica Cross-Validation\n")

print(f"Acierto con atributos originales para {at_objetivo} : {r_logistica(atributos, objetivo, t_size, 10, True)}\n")
print(f"Acierto con todos los atributos para {at_objetivo} : {r_logistica(atributos, objetivo, t_size, 10, True)}\n")
print(f"Acierto con los atributos relacionales solamente, para {at_objetivo}: ",r_logistica(atributos_total[["Centralidad_grado", "Centralidad_intermedia", "Centralidad_katz", "Grados", "Clusters"]],objetivo, t_size, 10, True),"\n")

for i in range(0,len(datos)):
    dat_usado = datos[i]
    at_nuevo = datos[i].columns[7]
    print(f"Acierto con atributos originales  mas",at_nuevo,"para",at_objetivo,":",r_logistica(dat_usado, objetivo, t_size, 10, True),"\n") 


#Árboles de decisión
print("\nArboles de decisión\n")

print(f"Acierto con atributos originales para {at_objetivo} : {arboles_decision(atributos, objetivo, t_size)[0]}\n")
print(f"Acierto con todos los atributos para {at_objetivo} : {arboles_decision(atributos, objetivo, t_size)[0]}\n")
print(f"Acierto con los atributos relacionales solamente, para {at_objetivo}: ",arboles_decision(atributos_total[["Centralidad_grado", "Centralidad_intermedia", "Centralidad_katz", "Grados", "Clusters"]],objetivo, t_size)[0],"\n")

for i in range(0,len(datos)):
    dat_usado = datos[i]
    at_nuevo = datos[i].columns[7]
    print(f"Acierto con atributos originales  mas",at_nuevo,"para",at_objetivo,":",arboles_decision(dat_usado, objetivo, t_size)[0],"\n")
    if(i == 2):
        tree.plot_tree(arboles_decision(dat_usado, objetivo, t_size)[1],class_names=True,filled=True,
                       feature_names=["Age","Birthplace","Residence","Arrests","Convictions","Music","Ranking","Centralidad_katz"])


#KNN
print("\nKNN - Euclidea\n")

print(f"Acierto con atributos originales para {at_objetivo} : {knn(atributos_total, objetivo)}\n")
print(f"Acierto con todos los atributos para {at_objetivo} : {knn(atributos_total, objetivo)}\n")
print(f"Acierto con los atributos relacionales solamente, para {at_objetivo}: ",knn(atributos_total[["Centralidad_grado", "Centralidad_intermedia", "Centralidad_katz", "Grados", "Clusters"]],objetivo),"\n")

for i in range(0,len(datos)):
    dat_usado = datos[i]
    at_nuevo = datos[i].columns[7]
    print(f"Acierto con atributos originales  mas",at_nuevo,"para",at_objetivo,":",knn(dat_usado, objetivo),"\n")