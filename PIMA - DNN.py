"""
Autor: Przemysław Scharmach
Skorzystano z ogólnodostępnego zestawu danych Pima Indiands Diabetes Dataset.

Opis problemu: Stworzenie sieci neuronowej na bazie architektury typu DNN
(Deep Neural Network -sieć z wieloma warstwami między warstwą wejściową i wyjściową.
Sieci DNN są zwykle sieciami z wyprzedzeniem, w których dane przepływają z 
warstwy wejściowej do warstwy wyjściowej bez zapętlenia. 
Na początku DNN tworzy mapę wirtualnych neuronów i przypisuje losowe wartości 
liczbowe lub „wagi” połączeniom między nimi. Wagi i dane wejściowe są mnożone 
i zwracają wynik od 0 do 1. Jeśli sieć nie rozpoznała dokładnie określonego 
wzorca, algorytm dostosowałby wagi. W ten sposób algorytm może zwiększyć wpływ 
niektórych parametrów, dopóki nie określi prawidłowej manipulacji matematycznej, 
aby w pełni przetworzyć dane


Model parametryczny - ilosc próbek może zmniejszać wariancję
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
import os 
import random
import tensorflow.random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Czynnik losowosci
"""
Dla pełnej powtarzalnosci ustawiamy konkretny seed losowy, dzięki temu 
nie będzie rozstrzalu między wynikami. Sam 'random_state=23' widoczny w późniejszej 
częci kodu z jakiego powodu w moim przypadku nie wystarczał, szukałem więc
innego rozwiązania.
"""

seed_value = 23
os.environ['PYTHONHASHSEED']=str(seed_value)

random.seed(seed_value)
tensorflow.random.set_seed(seed_value)

# Wczytanie pliku
"""
W tej sekcji wczytujemy nasz plik CSV do zmiennej dataset.
"""

dataset = pd.read_csv("diabetes.csv")
 
dataset.info() 
"""
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
Pregnancies                 768 non-null int64
Glucose                     768 non-null int64
BloodPressure               768 non-null int64
SkinThickness               768 non-null int64
Insulin                     768 non-null int64
BMI                         768 non-null float64
DiabetesPedigreeFunction    768 non-null float64
Age                         768 non-null int64
Outcome                     768 non-null int64
"""

"""
Nie jest możliwe osiągnięcie wskaźnika masy ciała lub stężenia glukozy w osoczu 
równego 0 więc czyscimy nasz dateset przd budowaniem modelu.
Można to szybko osiągnąć, zastępując brakujące wartości wartością NaN, 
a następnie usuwając te wiersze ze zbioru danych. Zastosowanie tego zabiegu
nie zmieniając niczego więcej poskutkowało zwiększeniem dokładnosci o 1.16%
"""
dataset[['Glucose','BMI']] = dataset[['Glucose','BMI']].replace(0, np.NaN)
dataset.dropna(inplace=True)


# Przypisanie atrybutów do zmiennych 
"""
Pod zmienną X wskazujemy nasze podstawowe 
atrybuty, natomiast pod zmienną Y kryje się atrybut przypisania do klasy.
"""

X = dataset.drop(['Outcome'], axis=1)
Y = dataset['Outcome']


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, 
                                                    random_state=23)

# Tworzenie modelu
"""
Każda warstwa może zastosować dowolną funkcję do poprzedniej warstwy.
Zadaniem warstw ukrytych jest przekształcenie danych wejściowych w coś, z 
czego warstwa wyjściowa może korzystać. Warstwa wyjściowa przekształca 
aktywacje warstwy ukrytej w dowolną skalę, w jakiej chcemy aby wyjście było.

Funkcja Softmax to funkcja aktywacji, która zamienia liczby w 
prawdopodobieństwa, które sumują się. Generuje wektor, który reprezentuje
rozkłady prawdopodobieństwa - listy potencjalnych wyników.
"""

model = Sequential()

# Zmienne pomocnicze
iterations = 1000
hold_prob = 0.0

# Warstaw wejsciowa
"""
Jeśli chodzi o liczbę neuronów wchodzących w skład warstwy wejsciowej,
parametr ten jest określany całkowicie i jednoznacznie, gdy znasz kształt 
danych treningowych. W szczególności liczba neuronów wchodzących w skład tej 
warstwy jest równa liczbie cech (kolumn) w danych. W naszym przypadku mamy 8
kolum.
"""

model.add(Dense(8, input_dim=8, activation='relu'))

# Warstwy ukryte
"""
Przyjmujemy praktykę:
"Liczba ukrytych neuronów powinna wynosic około 2/3 wielkości warstwy wejściowej plus 
rozmiar warstwy wyjściowej, stąd ustawiamy 8 neuronów (7.28 zaokrąglone do 8) 
na każdą warstwę ukrytą aby uniknąć przetrenowania.

Warstwa Dropout losowo ustawia jednostki wejściowe na 0 na każdym kroku 
podczas treningu, co pomaga zapobiegać nadmiernemu dopasowaniu. 
Wejścia nie ustawione na 0 są skalowane w górę tak, że suma wszystkich 
wejść pozostaje niezmieniona.
"""

model.add(Dense(8, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(hold_prob))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(hold_prob))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(hold_prob))

# Warstwa 
"""
W przypadku Softmax warstwa wyjsciowa ma jeden węzeł na każdą etykietę klasy
w modelu. Stąd tutaj wartosc "2".
"""

model.add(Dense(2, activation='softmax'))

# Kompilowanie
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Dopasowanie
history = model.fit(x_train,y_train, validation_data=(x_test, y_test), epochs=iterations, batch_size=70)
x_train.shape

# Wynik
"""
Ewaluacja. Wyswietlamy procentową skutecznosc.
"""

scores = model.evaluate(x_train,y_train)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Krzywa ROC 
"""Jest wykresem graficznym, który ilustruje zdolność 
diagnostyczną binarnego systemu klasyfikatora , gdy jego próg dyskryminacji
jest zmienny. Krzywa ROC jest tworzona przez wykreślenie wskaźnika prawdziwie 
dodatnich wyników (TPR) w stosunku do odsetka wyników fałszywie dodatnich 
(FPR) przy różnych ustawieniach progowych.
"""

curve_it=model.predict(x_test)
fpr, tpr, thresholds = roc_curve(y_test, curve_it[:, 1])
roc_auc = auc(fpr, tpr)
print("ROC AUC:", roc_auc)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Positive Test ROC')
plt.legend(loc="lower right")
plt.show()
