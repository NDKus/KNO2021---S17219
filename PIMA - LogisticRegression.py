"""
Autor: Przemysław Scharmach
Skorzystano z ogólnodostępnego zestawu danych Pima Indiands Diabetes Dataset.

Opis problemu: Wykorzystanie regresji logistycznej do klasyfikacji danych i
przewidywania prawdopodobieństwa.

Regresja logistyczna to kolejna technika zapożyczona przez uczenie maszynowe 
z dziedziny statystyki. Jest to podstawowa metoda rozwiązywania problemów z 
klasyfikacją binarną (problemy z dwiema wartościami klas).
Współczynniki algorytmu regresji logistycznej należy oszacować na podstawie 
danych szkoleniowych. Odbywa się to za pomocą oszacowania największej 
wiarygodności.

- Wysoki BIAS 
- Niski variance
- Raczej brak ryzyka overfittingu (model parmetryczny)
- Underfitting nie występuje (zaobserwowanoby to po wydajnoci w zestawie szkoleniowym)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Wczytanie pliku
"""
W tej sekcji wczytujemy nasz plik CSV do zmiennej dataset.
"""
data = pd.read_csv("diabetes.csv")


# Charakterystyka danych
data.info() 
"""
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
0 Pregnancies                 768 non-null int64
1 Glucose                     768 non-null int64
2 BloodPressure               768 non-null int64
3 SkinThickness               768 non-null int64
4 Insulin                     768 non-null int64
5 BMI                         768 non-null float64
6  DiabetesPedigreeFunction   768 non-null float64
7 Age                         768 non-null int64
8 Outcome                     768 non-null int64
"""

# Wizualizacja korelacji cech
def visualise(data):
    fig, ax = plt.subplots()
    ax.scatter(data.iloc[:,1].values, data.iloc[:,5].values)
    ax.set_title('Highly Correlated Features')
    ax.set_xlabel('Plasma glucose concentration')
    ax.set_ylabel('Body mass index')

# Wyswietlenie wykresu przed czyszczeniem
visualise(data)


"""
Nie jest możliwe osiągnięcie wskaźnika masy ciała lub stężenia glukozy w osoczu 
równego 0 więc czyscimy nasz dateset przd budowaniem modelu.
Można to szybko osiągnąć, zastępując brakujące wartości wartością NaN, 
a następnie usuwając te wiersze ze zbioru danych. Zastosowanie tego zabiegu
nie zmieniając niczego więcej poskutkowało zwiększeniem dokładnosci o 
"""

# Czyszczenie datasetu
data[['Glucose','BMI']] = data[['Glucose','BMI']].replace(0, np.NaN)
data.dropna(inplace=True)

# Wyswietlenie wykresu po czyszczeniu
visualise(data)

# Wyodrębnienie dwóch cech do budowy klasyfikatora regresji logistycznej
"""
Zdecydowalismy przy pomocy wizualizacji korelacji danych, iż aby zbudować nasz
klasyfikator regresji logistycznej wykorzystamy glukozę (index 1) oraz BMI
(index 5), więc wyodrębniamy te kolumny cech oraz ich wartosci docelowe.
"""
X = data[['Glucose','BMI']].values
y = data[['Outcome']].values

"""
Stosujemy skalowanie (już po wyizolowaniu kolumn) w celu osiągnięcia lepszych 
rezultatów uczenia. Stosujemy dopasowanie.
"""
sc = StandardScaler()
X = sc.fit_transform(X)

"""
Po zastosowaniu skalowania nasz dataset jest dwuwymiarową tablicą, w której
kolumny transformowano w taki sposób, że srednia wartosć dystrybunaty jest 
równa 0, a odchylenie standardowe jest równe 1.
"""
mean = np.mean(X, axis=0)
print('Mean: (%d, %d)' % (mean[0], mean[1]))
standard_deviation = np.std(X, axis=0)
print('Standard deviation: (%d, %d)' % (standard_deviation[0], standard_deviation[1]))

"""
Teraz standardowo podzielimy zbiór danych na zbiór uczący i zestaw testowy, 
który będzie przydatny podczas oceny wydajności modelu. 
Podczas dzielenia zbioru danych zostanie zastosowany ponownie podział 70:30
"""
# Podział na dane treningowe i testowe

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 23)

"""
Gdy mamy już nasz zestaw szkoleniowy i testowy, możemy zdefiniować model 
LogisticRegression i dopasować go do naszych danych szkoleniowych. 
Po przeszkoleniu model można następnie wykorzystać do prognozowania.
"""
# Wskazanie modelu i dopasowanie

model = LogisticRegression()

model.fit(X_train, y_train.ravel())
y_pred = model.predict(X_test)

"""
Confusion matrix jest podsumowaniem wyników predykcji dla danego problemu 
klasyfikacyjnego. Pokazuje liczbę poprawnych i 
niepoprawnych prognoz w podziale na każdą klasę.

W tym przypadku liczba prawdziwych pozytywów wyniosła 127, 
a prawdziwie negatywnych 40. Oznacza to, że wykonaliśmy łącznie 167
poprawnych przewidywań ze 226 (~ 74%). Macierz pozwala nam również 
przewidzieć dwie dodatkowe statystyki, które są dobre do oceny modelu, 
a mianowicie precyzję i powtarzalność.

Precyzja to stosunek tp / (tp + fp), gdzie tp to liczba prawdziwych trafień, 
a fp to liczba fałszywych trafień. "Precission" to zdolność 
klasyfikatora do nie oznaczania pozytywnej próbki jako negatywnej.

"Recall" to stosunek tp / (tp + fn), gdzie tp to liczba prawdziwych 
wyników pozytywnych, a fn liczba fałszywie negatywnych. To zdolność 
klasyfikatora do znalezienia wszystkich pozytywnych próbek.

Precision-Recall jest doskonałą miarą sukcesu predykcji dla niezbalansowanych 
klas. W wyszukiwaniu informacji precyzja jest miarą trafności wyników, 
podczas gdy przypominanie jest miarą liczby zwracanych naprawdę 
trafnych wyników.
"""
# Utworzenie i wyswietlenie macierzy "confusion matrix"
cm = confusion_matrix(y_test, y_pred)
print(cm)

def precision_recall(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)    
    tp = cm[0,0]
    fp = cm[0,1]
    fn = cm[1,0]
    prec = tp / (tp+fp)
    rec = tp / (tp+fn)    
    return prec, rec

# Weryfikacja precyzji oraz powtarzalnosci przy pomocy metody precision_recall
    

precision, recall = precision_recall(y_test, y_pred)
print('Precision: %f Recall %f' % (precision, recall))

