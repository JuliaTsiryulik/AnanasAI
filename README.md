[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=10439323&assignment_repo_type=AssignmentRepo)
# Лабораторная работа по курсу "Искусственный интеллект"
# Создание своего нейросетевого фреймворка

### Студентка: 

| ФИО       | Роль в проекте                     | Оценка       |
|-----------|------------------------------------|--------------|
| Копытина Юлия | Программировал |       |
| Копытина Юлия | Программировал оптимизаторы |      |
| Копытина Юлия | Писала отчёт |          |

> Документацию можно писать прямо в этом файле, удалив задание, но оставив табличку с фамилиями студентов. Необходимо разместить решение в репозитории Github Classroom, но крайне рекомендую также сделать для этого отдельный открытый github-репозиторий (можно путём fork-инга).

### AnanasAI

Фреймворк включает в себя следующие возможности:

1. Создание многослойной нейросети перечислением слоёв,
2. Удобный набор функций для работы с данными и датасетами,
3. Различные алгоритмы оптимизации, передаточные функции и функции потерь для решения задач классификации и регрессии,
4. Обучение нейросети "в несколько строк", при этом с гибкой возможностью конфигурирования,
5. Несколько примеров использования нейросети на классических задачах (MNIST, Iris).

Для запуска фреймворка необходимо скачать этот github.

Затем в вашей командной строке выполнить команду:

```
>pip install /path/to/dist/ananas_ai_lib-0.1.0-py3-none-any.whl
```

Теперь вы можете использовать данную библиотеку следующим образом:

```
from ananas_ai_lib.ananas_framework import *
```

или

```
import ananas_ai_lib.ananas_framework as ananas
```

#### Параметры:

***add_layers(layers, loss_function, optimizer=SGD(), learning_rate=0.02, 
           epochs_count=20, classification = True, regression = False)***
           
Происходит подготовка модели.
<br/><br/>**layers:**
    <br/>&emsp;Слои нейронной сети, необходимо указывать в качестве списка. 
    <br/>&emsp;Сумматор:
        <br/>&emsp;&emsp;**Linear(n_in, n_out):**
            <br/>&emsp;&emsp;&emsp;**n_in: int**
                <br/>&emsp;&emsp;&emsp;&emsp;Количество входов.
            <br/>&emsp;&emsp;&emsp;**n_out: int**
                <br/>&emsp;&emsp;&emsp;&emsp;Количество выходов.
    <br/>&emsp;Доступны следующие функции активации:
        <br/>&emsp;&emsp;**ReLU(),
        <br/>&emsp;&emsp;Tanh(),
        <br/>&emsp;&emsp;Sigmoid(),
        <br/>&emsp;&emsp;Softmax().**       
<br/>**loss_function:**
    <br/>&emsp;Функция потерь.
    <br/>&emsp;Доступны следующие функции потерь:
        <br/>&emsp;&emsp;**BinaryCrossEntropy(),
        <br/>&emsp;&emsp;CrossEntropyLoss(),
        <br/>&emsp;&emsp;MeanSquaredError().**
<br/><br/>**optimizer: default=SGD()**
    <br/>&emsp;Алгоритм оптимизации.
    <br/>&emsp;Доступны следующие алгоритмы оптимизации:
        <br/>&emsp;&emsp;**SGD(): learning_rate=0.02,
        <br/>&emsp;&emsp;MomentumSGD(): learning_rate=0.2,
        <br/>&emsp;&emsp;Adam(): learning_rate=0.99.**
<br/><br/>**learning_rate: float, default=0.02**
    <br/>&emsp;Скорость обучения.  
<br/>**epochs_count: int, default=20**
    <br/>&emsp;Количество эпох.   
<br/>**classification: bool, default=True**
    <br/>&emsp;Решение задачи классификации.    
<br/>**regression: bool, default=False**
    <br/>&emsp;Решение задачи регрессии.

<br/>***train(X_train, Y_train)***

Тренирует модель.
<br/><br/>**X_train:**
    <br/>&emsp;Массив признаков обучающей выборки.
<br/>**Y_train:**
    <br/>&emsp;Массив меток обучающей выборки.

<br/>***predict(X_test, Y_test)***

Возвращает предсказанные значения меток и точность (классификация) или значение функции потерь (регрессия).
<br/><br/>**X_test:**
    <br/>&emsp;Массив признаков тестовой выборки.
<br/>**Y_test:**
    <br/>&emsp;Массив меток тестовой выборки.


*Пример (бинарная классификация):*

```
import ananas_ai_lib.ananas_framework as ananas

model = ananas.NeuralFramework()
model.add_layers(layers = [ananas.Linear(2, 5), ananas.ReLU(), 
                           ananas.Linear(5, 3), ananas.Tanh(), 
                           ananas.Linear(3, 2), ananas.Sigmoid()],
                 loss_function = ananas.BinaryCrossEntropy(), 
                 optimizer = ananas.Adam())
model.train(X_train, Y_train)
pred_vals, acc = model.predict(X_test, Y_test)
```

Больше примеров можно увидеть в файле "LR-1. FrameWork.ipynb"
