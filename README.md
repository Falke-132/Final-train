# Final-train
Обнаружение фишинга - Пайплайн обучения модели

Проект для подготовки данных и обучения модели обнаружения фишинговых веб-страниц на основе 87 признаков URL и контента. В качестве алгоритма используется градиентный бустинг (LightGBM).

Бейзлайн и датасет взяты из научной статьи: [Hannousse, A., & Yahiouche, S. (2020). "Towards benchmark datasets for machine learning based website phishing detection".](https://arxiv.org/pdf/2010.12847)

⚠️**ВНИМАНИЕ**: Файл data/dataset_phishing.csv содержит сырые URL-адреса реальных фишинговых сайтов.
Не открывайте эти ссылки в браузере. Они предоставлены исключительно для обучения модели машинного обучения.
Будьте осторожны.


## Результаты

В ходе проекта удалось превзойти метрики, описанные в оригинальной статье, за счет использования современного алгоритма градиентного бустинга и подбора гиперпараметров.
Метрика	Оригинальная статья (2020)	Модель проекта (LightGBM)
Accuracy	0.9683	0.9707
Macro F1-score	0.9680	0.9707
ROC-AUC	-	0.9948

### Структура проекта
```text
phishing-training/
│
├── data/                         # Папка для данных
│   └── dataset_phishing.csv       # Исходный сырой датасет
│
├── notebooks/                    # Jupyter тетради
│   └── 01_eda_and_model.ipynb    # EDA, поиск гиперпараметров
│
├── prepare_data.py               # Скрипт очистки данных и выделения демо
├── train.py                      # Скрипт обучения модели
├── requirements.txt              # Зависимости
└── README.md
```

## Установка и запуск

### 1. Требования
- Python 3.9+
- Рекомендуется использовать виртуальное окружение (`venv`)

### 2. Установка зависимостей
```bash
git clone https://github.com/Falke-132/Final-train.git
cd Final-train

python -m venv venv
source venv/bin/activate  # Для Linux/Mac
venv\Scripts\activate     # Для Windows

pip install --upgrade pip
pip install -r requirements.txt
``` 
 
### 3. Воспроизведение пайплайна (Обучение с нуля)

#### Шаг 3.1. Подготовка данных
Скрипт берет сырой датасет, кодирует целевую переменную, удаляет константные и шумовые признаки, а также вырезает 10 строк для демо-режима.
```bash
python prepare_data.py \
    --input data/dataset_phishing.csv \
    --output-csv data/clean_phishing.csv \
    --output-demo data/demo_urls.json
``` 

#### Шаг 3.2. Обучение модели
Скрипт обучает LightGBM и сохраняет модель, метрики и список фичей в папку models/.
``` bash
python train.py \
    --input data/clean_phishing.csv \
    --output models
```
