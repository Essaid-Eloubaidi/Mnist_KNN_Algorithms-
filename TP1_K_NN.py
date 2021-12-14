# ce travail est realise par ESSAID EL-OUBAIDI 
# INSA de Toulouse 


# bibliotheque
from sklearn.datasets import fetch_openml
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score 




#Telechargement de datasets
mnist = fetch_openml('mnist_784',as_frame=False)




#développement d'un modèle 4_NN avec 70% des données d'entraînement et 30% des données de test
data=mnist.data
target=mnist.target
xtrain, xtest, ytrain, ytest = train_test_split(data, target,train_size=0.7)
neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(xtrain, ytrain)
print("Lancement de la prédiction ...")
prediction=neigh.predict(xtest[0:1])
prediction_proba=neigh.predict_proba(xtest[0:1])
score=neigh.score(xtest, ytest)
print(prediction)
print("La probabilité sur chaque classe : ", prediction_proba)
print("le score global de model est  : ", score)





#Développement d'un modèle 10_NN avec 80% des données d'entraînement et 20% des données de test
data=mnist.data
target=mnist.target
xtrain, xtest, ytrain, ytest = train_test_split(data, target,train_size=0.8)
neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(xtrain, ytrain)
#prediction d'une image 
predict_image=neigh.predict(xtest[7:8])
print(predict_image)
# classe de l'image 
print(ytest[7:8])
if(predict_image == ytest[7:8]):
    print("prediction avec succes")
else :
     print("prediction echoue")
score=neigh.score(xtest[7:8],ytest[7:8])
print(" le score sur cet echanntillons est : ", score)
scores=neigh.score(xtest,ytest)
print(" le score globale  est : ", scores)

# l'erreur de modele 
erreur = 1-neigh.score(xtest,ytest)
print("erreur est : {}".format(erreur))




#la valeur de k optimale avec une boucle 
erreur=[]
for k in range (2,15):
    neigh = KNeighborsClassifier(k)
    erreur.append(100*(1-neigh.fit(xtrain, ytrain).neigh.score(xtest,ytest)))
    
plt.plot(range(2,15),erreur, '-o')
plt.title("l'erreur en fonction de diffenets valeurs de k")
plt.xlabel("valeurs de K")
plt.ylabel("l'erreur")
plt.show()





#la valeur de k optimale avec la methode de KFold
KF=KFold(10,shuffle=True)
scores= []
for k in range (2,15):
    score=cross_val_score(KNeighborsClassifier(K),xtest,ytest,cv=KF,scoring='accuracy').mean()
    scores.append(score)
plt.plot(range(2,15),erreur, '-o')
plt.title("Score en fonction de diffenets valeurs de k")
plt.xlabel("valeurs de K")
plt.ylabel("scores")
plt.show()




# teter les differentes distances 
for i in range(0,3):
  distance=["manhattan","euclidean","minkowski"]
  neigh=KNeighborsClassifier(10,p=)
  neigh.fit(xtest,ytest)
  prediction=neigh.predict(xtest)
  score=neigh.score(xtest,ytest)7
  print(" le type de distance est : {} , le score est : {} ".format(distance[i],score))
  
  
  
  
 
#Tester les valeurs de n_jobs : 1 et -1 
for i in [-1,1]:
  neigh=KNeighborsClassifier(5,n_jobs=i)
  neigh.fit(xtest,ytest)
  start=time.time()
  prediction=neigh.predict(xtest)
  stop=time.time()
  score=neigh.score(xtest,ytest)
  print(" le n_jons est : {} , le temps de prediction est : {} ".format(i,str(stop-start)))
    
