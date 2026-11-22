# mle-analiz-loyalnosti-polzovatelej

# Проект анализа лояльности пользователей яндекс афиши

## Ссылка на github

```
https://github.com/isaev-andrew-ds/mle-analiz-loyalnosti-polzovatelej
```

## Запуск

1. В github нажмите `clone` -> `ssh`/`https` *(любой из них)* -> скопируйте `url` *(кнопка копирования)*

2. Склонируйте репозиторий 

```bash
git clone {url}
```

3. Создайте среду выполнения python в папке проекта, например, `venv`:

```shell
python -m venv .venv
```

4. Активируйте созданную среду

```shell
.venv/Scripts/active
```

4. Установите зависимости из `requirements.txt` 

```shell
pip install -r requirements.txt
```

5. Создайте .env файл в корневой директории проекта, скопировав содержимое из `.env.example`

```bash
cp .env.example .env
```

6. В `.env` файле поставьте **свои** значения в переменные `DB_USER` и `DB_PWD` вместо значений `FIXME`
7. Теперь можно запустить главный файл `main.py` в терминале
```shell
python main.py
```

## Стек

- python
- pandas
- matplotlib
- seaborn
- numpy
- phik

## Структура проекта

```
|-data  -   Датасеты
|-src
|   |-processing.py -   класс анализа данных
|-main.py   -   главный файл запуска
|-.env.example  -   необходимые переменные окружения с примерами значений
```