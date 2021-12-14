# ce travail est realise par ESSAID EL-OUBAIDI 
# INSA de Toulouse 


# Les bibliotheques utilises 
from sklearn.datasets import fetch_openml
from sklearn import datasets 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score
import time


mnist = fetch_openml('mnist_784',as_frame=False)


data=mnist.data
target=mnist.target
xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.7)

# un modele MLP avec 50 reseaux de neurons 
hidden_layer_sizes = (50)
clf=MLPClassifier(hidden_layer_sizes)
clf.fit(xtrain, ytrain)
y_predict =  clf.predict(xtest)
print(y_predict)
print("la précision est : ", precision_score(ytest, y_predict, average='macro')) 


# Variation de nombre de couches de modele 
hidden_layer_sizes = (40,)*100
start=time.time()
predict=[]
loss_0_1 = []
for i in range(2,101):
     
    clf=MLPClassifier(hidden_layer_sizes[0:i] ,verbose=True)
    print("le nombre des couches est : " , len(hidden_layer_sizes[0:i]))
    clf.fit(xtrain, ytrain)
    y_predict = clf.predict(xtest)
    predict.append(precision_score(ytest, y_predict, average='macro'))
    loss_0_1.append(metrics.zero_one_loss(ytest, y_predict))
stop=time.time()   
print("le temps de l'apprentissage est :" + str(stop-start) + "s")

fig, axarr = plt.subplots(2, sharex=True, figsize=(10,10))
axarr[0].plot(range(100), predict)
axarr[0].set_ylabel('Precision') 
axarr[1].plot(range(100), loss_0_1)
axarr[1].set_ylabel('Zero-to-one Loss')   



# Test 
print("Prediction de l'image 4  : l'emage testé est  : ", ytest[3], "    sa classe Predite est ", y_predict[3] )



# Etude de la performance de 5 modele MLP
clf_1 = MLPClassifier(hidden_layer_sizes=(300))
clf_3 = MLPClassifier(hidden_layer_sizes=(30, 100, 60))
clf_5 = MLPClassifier(hidden_layer_sizes=(40, 200, 70, 100, 10))
clf_7 = MLPClassifier(hidden_layer_sizes=(300, 20, 280, 50, 90, 7, 10))
clf_10 = MLPClassifier(hidden_layer_sizes=(30, 60,3,14,289, 180, ,7,210, 240, 270))

# La fonction de performance 
Score =[]
Precision = []
zero_one_loss = []

def performance_classification(clf):

    start =time.time()
    clf.fit(xtrain, ytrain)
    end = time.time()
    T= end-start
    predict = clf.predict(xtest)

    score = clf.score(xtest,ytest)
    precision =  precision_score(ytest, predict,  average='macro')
    loss0_1 =    zero_one_loss(ytest, predict)

    Score.append(score)
    Precision.append(precision)
    zero_one_loss.append(loss0_1)

    print(" le score de modèle est  = ", score*100, ", precision =", precision*100, )
    print(" le temps d'apprentissage est  ", str(T), "(s) ." )
# Test 
performance_classification(clf_5  
    
    
  
    
# les architectures de test 
arch1 = (300)
arch2 = (30, 100, 60)
arch3 = (40, 200, 70, 100, 10)
arch4 = (300, 20, 280, 50, 90, 7, 10)
arch5 =(30, 60,3,14,289, 180,7,210, 240, 270)
  
#  Les performances des algorithmes d’optimisation L-BFGS, SGD et Adam.
Score = []
Precision = []
Loss = []
def Solver(arch, solv):

    clf = MLPClassifier(hidden_layer_sizes = arch, solver = solv)
    start = time.time()
    clf.fit(xtrain, ytrain)
    end = time.time()
    prediction = clf.predict(xtest)
    score = clf.score(xtest, ytest)
    precision = precision_score(ytest, prediction, average = 'macro')
    loss0_1 = zero_one_loss(ytest, prediction)
    T = end - start
    Score.append(score)
    Precision.append(precision)
    Loss.append(loss0_1)
    print("Pour le  solver : ", solv)
    print(" score = ", score * 100, "%, precision = ", precision * 100, "%")
    print(" le temps d'apprentissage est  ", str(T), " s ." )

# Test 
print("pour l'architecture 1 : ")
for i in ('lbfgs', 'sgd', 'adam'):
    Solver(arch1, i)
    
    
    
    
# Les performances des fonctions d’activation : identity, logistic, tanh et relu
Score = []
Precision = []
Loss = []
def activation_function(arch,activation):

    clf = MLPClassifier(hidden_layer_sizes = arch, activation = activation)
    start = time.time()
    clf.fit(xtrain, ytrain)
    end = time.time()
    prediction = clf.predict(xtest)
    score = clf.score(xtest, ytest)
    precision = precision_score(ytest, prediction, average = 'macro')
    loss0_1 = zero_one_loss(ytest, prediction)
    T = end - start
    Score.append(score)
    Precision.append(precision)
    Loss.append(loss0_1)
    print("Pour la fonction d'activation  : ", activation)
    print(" score = ", score * 100, "%, precision = ", precision * 100, "%")
    print(" le temps d'apprentissage est  ", str(T), " s ." )

# Test 
print("pour l'architecture 5 : ")
for i in ('identity', 'logistic', 'tanh', 'relu'):
    activation_function(arch5,i)
    
    
    
# La valeur optimale de la régularisation L2 (paramètre α).    
alpha = np.logspace(-5, 3, 5)
Score = []
Precision = []
Loss = []
def test_alpha(arch,a):
    

    clf = MLPClassifier(hidden_layer_sizes = arch1, alpha=0.00001, activation='relu', solver="adam")
    start = time.time()
    clf.fit(xtrain, ytrain)
    end = time.time()
    prediction = clf.predict(xtest)
    score = clf.score(xtest, ytest)
    precision = precision_score(ytest, prediction, average = 'macro')
    loss0_1 = zero_one_loss(ytest, prediction)
    T = end - start
    Score.append(score)
    Precision.append(precision)
    Loss.append(loss0_1)
    print("Pour la valeur de alpha   : ", a)
    print(" score = ", score * 100, "%, precision = ", precision * 100, "%")
    print(" le temps d'apprentissage est  ", str(T), " s ." )
    
    
#Test 
print("pour l'architecture 4 : ")
for i in alpha:
    test_alpha(arch4, i)
    




# Conclusion : avec les etudes et les essaies effectues sur les differents parametres de modele,
# nous concluons que le meilleur medele pour cette datasets est le suivant :

clf = MLPClassifier(hidden_layer_sizes = arch1, alpha=0.00001, activation='relu', solver="adam")



















