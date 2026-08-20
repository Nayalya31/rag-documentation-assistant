# RAG-ассистент по документации

Ассистент отвечает на вопросы по внутреннему мануалу бухгалтерской компании. Он ищет релевантные фрагменты в базе документов (RAG), генерирует ответ через OpenAI и может работать в консоли или в Telegram.

## Возможности

- Поиск по документации с векторным хранилищем **ChromaDB** и эмбеддингами OpenAI
- Генерация ответов моделью **gpt-4o-mini** только на основе найденного контекста
- Кеш вопрос–ответ в SQLite
- Консольный чат
- Telegram-бот с кнопками меню и примерами вопросов
- Оценка качества через **RAGAS**: Faithfulness, Context Precision, Answer Relevancy

## Структура проекта

```
.
├── requirements.txt
├── README.md
└── assistant_api/
    ├── app.py              # консольный ассистент
    ├── telegram_bot.py     # Telegram-бот
    ├── rag_pipeline.py     # RAG: кеш → поиск → LLM
    ├── vector_store.py     # ChromaDB и chunking
    ├── cache.py            # SQLite-кеш
    ├── evaluate_ragas.py   # оценка качества
    ├── data/docs.txt       # база документов
    └── .env.example        # шаблон ключей
```

## Требования

- Python 3.11
- ключ [OpenAI API](https://platform.openai.com/api-keys)
- для бота: токен от [@BotFather](https://t.me/BotFather)

## Установка

```bash
python3.11 -m venv venv_py311
source venv_py311/bin/activate          # Windows: venv_py311\Scripts\activate
pip install -r requirements.txt
```

Скопируйте шаблон переменных окружения и подставьте свои ключи:

```bash
cp assistant_api/.env.example assistant_api/.env
```

В `assistant_api/.env`:

```env
OPENAI_API_KEY=sk-...
TELEGRAM_BOT_TOKEN=123456789:AA...
```

Файл `.env` в репозиторий не попадает.

## Запуск

Все команды ниже выполняйте из папки `assistant_api` при активированном окружении.

### Консольный ассистент

```bash
cd assistant_api
python app.py
```

Команды в чате: вопрос текстом, `stats`, `clear`, `exit`.

### Telegram-бот

```bash
cd assistant_api
python telegram_bot.py
```

Откройте бота в Telegram и отправьте `/start`.

Кнопки меню:

- **Задать вопрос** — подсказка, как спросить
- **Примеры** — готовые вопросы из мануала
- **Статистика** — хранилище и кеш
- **Очистить кеш** — с подтверждением
- **Помощь** — справка

Любой обычный текст тоже уходит в RAG.

### Оценка качества (RAGAS)

```bash
cd assistant_api
python evaluate_ragas.py
```

Скрипт задаёт тестовые вопросы, считает метрики и выводит средний балл. Занимает 1–2 минуты и использует OpenAI API.

## Как это работает

1. Документ `data/docs.txt` разбивается на чанки и сохраняется в ChromaDB.
2. Вопрос пользователя ищется по эмбеддингам (`text-embedding-3-small`).
3. Найденный контекст подставляется в промпт.
4. Модель отвечает только по этому контексту.
5. Пара вопрос–ответ сохраняется в кеш.

## Лицензия

Учебный проект. Используйте свободно в образовательных целях.
