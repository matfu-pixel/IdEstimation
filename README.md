# Сравнение методов оценки внутренней размерности данных

1. Проведенные эксперименты находятся в папке `/experiments`
2. Код модуля с реализаций методов находится в папке `/src`
3. Отчет о проделанной работе `IdEstimation.pdf`

### Запуск

```
import estimators  
...
cpca = estimators.cPCA()  
cpca.fit(data)  
print(cpca.dimension_)  
fisherS = estimators.FisherS()  
fisherS.fit(data)  
print(fisherS.dimension_)
```
