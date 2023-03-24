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

Для 
**add_layers(layers, loss_function, optimizer=SGD(), learning_rate=0.02, 
           epochs_count=20, classification = True, regression = False)**
Происходит подготовка модели.

layers:
    Слои нейронной сети, необходимо указывать в качестве списка. 
    Сумматор:
        Linear(n_in, n_out):
            n_in: int
                Количество входов.
            n_out: int
                Количество выходов.
    Доступны следующие функции активации:
        ReLU(),
        Tanh(),
        Sigmoid(),
        Softmax().
loss_function:
    Функция потерь.
    Доступны следующие функции потерь:
        BinaryCrossEntropy(),
        CrossEntropyLoss(),
        MeanSquaredError().

optimizer: default=SGD()
    Алгоритм оптимизации.
    Доступны следующие алгоритмы оптимизации:
        SGD(): learning_rate=0.02,
        MomentumSGD(): learning_rate=0.2,
        Adam(): learning_rate=0.99.

learning_rate: float, default=0.02
    Скорость обучения.
epochs_count: int, default=20
    Количество эпох.
classification: bool, default=True
    Решение задачи классификации.
regression: bool, default=False
    Решение задачи регрессии.

**train(X_train, Y_train)**
Тренирует модель.

X_train:
    Массив признаков обучающей выборки.
Y_train:
    Массив меток обучающей выборки.

**predict(X_test, Y_test)**
Возвращает предсказанные значения меток и точность (классификация) или значение функции потерь (регрессия).

X_test:
    Массив признаков тестовой выборки.
Y_test:
    Массив меток тестовой выборки.


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
