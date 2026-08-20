"""
Telegram-бот для RAG-ассистента по документации.
Кнопки меню, команды и ответы по базе документов.
"""

import asyncio
import html
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from rag_pipeline import RAGPipeline

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

for env_path in (BASE_DIR / ".env", BASE_DIR.parent / ".env"):
    if env_path.exists():
        load_dotenv(env_path)
        break
else:
    load_dotenv()

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("telegram_bot")

BTN_ASK = "❓ Задать вопрос"
BTN_STATS = "📊 Статистика"
BTN_EXAMPLES = "📚 Примеры"
BTN_CLEAR = "🗑 Очистить кеш"
BTN_HELP = "ℹ️ Помощь"

MENU_BUTTONS = [
    [BTN_ASK, BTN_EXAMPLES],
    [BTN_STATS, BTN_CLEAR],
    [BTN_HELP],
]

EXAMPLE_QUESTIONS = [
    "Что такое бухгалтерская услуга?",
    "Для кого подходят бухгалтерские услуги?",
    "Что такое ЭДО?",
    "Что такое электронная подпись?",
    "Что такое электронный ключ?",
]

TELEGRAM_LIMIT = 4000


def main_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        MENU_BUTTONS,
        resize_keyboard=True,
        input_field_placeholder="Введите вопрос по документации...",
    )


def examples_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(text=question, callback_data=f"example:{index}")]
            for index, question in enumerate(EXAMPLE_QUESTIONS)
        ]
    )


def clear_confirm_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("✅ Да, очистить", callback_data="clear:yes"),
                InlineKeyboardButton("❌ Отмена", callback_data="clear:no"),
            ]
        ]
    )


def get_pipeline(context: ContextTypes.DEFAULT_TYPE) -> RAGPipeline:
    return context.application.bot_data["pipeline"]


def format_answer(result: dict) -> str:
    if result["from_cache"]:
        source = "💾 Источник: кеш"
        if result.get("cached_at"):
            source += f"\n🕒 Сохранено: {result['cached_at']}"
    else:
        model = result.get("model", "LLM")
        docs_count = len(result.get("context_docs") or [])
        source = f"🌐 Источник: OpenAI ({model})\n📄 Документов: {docs_count}"

    text = (
        f"❓ <b>Вопрос</b>\n{html.escape(result['query'])}\n\n"
        f"{source}\n\n"
        f"💬 <b>Ответ</b>\n{html.escape(result['answer'])}"
    )

    if not result["from_cache"] and result.get("context_docs"):
        previews = []
        for i, doc in enumerate(result["context_docs"][:2], 1):
            preview = doc["text"][:180].strip()
            if len(doc["text"]) > 180:
                preview += "..."
            previews.append(f"{i}. {html.escape(preview)}")
        text += "\n\n📚 <b>Контекст</b>\n" + "\n\n".join(previews)

    if len(text) > TELEGRAM_LIMIT:
        text = text[: TELEGRAM_LIMIT - 20] + "\n\n…"
    return text


def format_stats(pipeline: RAGPipeline) -> str:
    stats = pipeline.get_stats()
    cache = stats["cache"]
    store = stats["vector_store"]
    oldest = cache["oldest_entry"] or "—"
    newest = cache["newest_entry"] or "—"
    return (
        "📊 <b>Статистика системы</b>\n\n"
        "🗂 <b>Векторное хранилище</b>\n"
        f"• Коллекция: {store['name']}\n"
        f"• Документов: {store['count']}\n"
        f"• Директория: {store['persist_directory']}\n\n"
        "💾 <b>Кеш</b>\n"
        f"• Записей: {cache['total_entries']}\n"
        f"• Размер БД: {cache['db_size_mb']:.2f} MB\n"
        f"• Первая запись: {oldest}\n"
        f"• Последняя запись: {newest}\n\n"
        f"🤖 Модель: {stats['model']}\n"
        f"🌐 Режим: {stats['mode']}"
    )


HELP_TEXT = (
    "ℹ️ <b>Как пользоваться ботом</b>\n\n"
    "Напишите вопрос по документации бухгалтерской компании — "
    "я найду нужный фрагмент и отвечу.\n\n"
    "<b>Кнопки меню</b>\n"
    f"• {BTN_ASK} — подсказка, как задать вопрос\n"
    f"• {BTN_EXAMPLES} — готовые вопросы из мануала\n"
    f"• {BTN_STATS} — статистика хранилища и кеша\n"
    f"• {BTN_CLEAR} — очистка кеша ответов\n"
    f"• {BTN_HELP} — эта справка\n\n"
    "<b>Команды</b>\n"
    "/start — главное меню\n"
    "/help — справка\n"
    "/examples — примеры вопросов\n"
    "/stats — статистика\n"
    "/clear — очистить кеш"
)


async def answer_question(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    question: str,
    *,
    via_callback: bool = False,
) -> None:
    question = (question or "").strip()
    if not question:
        message = "⚠️ Пожалуйста, введите вопрос."
        if via_callback and update.callback_query:
            await update.callback_query.message.reply_text(message, reply_markup=main_keyboard())
        elif update.message:
            await update.message.reply_text(message, reply_markup=main_keyboard())
        return

    chat = update.effective_chat
    status_message = None
    if via_callback and update.callback_query:
        await update.callback_query.answer()
        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
        status_message = await update.callback_query.message.reply_text("⏳ Ищу ответ в документации...")
    else:
        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
        status_message = await update.message.reply_text("⏳ Ищу ответ в документации...")

    pipeline = get_pipeline(context)
    try:
        result = await asyncio.to_thread(pipeline.query, question)
        text = format_answer(result)
        await status_message.edit_text(text, parse_mode=ParseMode.HTML)
    except Exception:
        logger.exception("Ошибка при обработке вопроса")
        await status_message.edit_text(
            "❌ Не удалось получить ответ. Попробуйте ещё раз через минуту."
        )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "👋 Привет! Я RAG-ассистент по документации бухгалтерской компании.\n\n"
        "Выберите действие в меню или просто напишите вопрос.",
        reply_markup=main_keyboard(),
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        HELP_TEXT,
        parse_mode=ParseMode.HTML,
        reply_markup=main_keyboard(),
    )


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = format_stats(get_pipeline(context))
    await update.message.reply_text(
        text,
        parse_mode=ParseMode.HTML,
        reply_markup=main_keyboard(),
    )


async def cmd_examples(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📚 Выберите пример вопроса:",
        reply_markup=examples_keyboard(),
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "⚠️ Очистить кеш ответов? Это действие нельзя отменить.",
        reply_markup=clear_confirm_keyboard(),
    )


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    data = query.data or ""

    if data.startswith("example:"):
        index = int(data.split(":", 1)[1])
        question = EXAMPLE_QUESTIONS[index]
        await answer_question(update, context, question, via_callback=True)
        return

    if data == "clear:yes":
        get_pipeline(context).cache.clear()
        await query.answer("Кеш очищен")
        await query.edit_message_text("✅ Кеш ответов очищен.")
        return

    if data == "clear:no":
        await query.answer("Отменено")
        await query.edit_message_text("Ок, кеш не трогали.")
        return

    await query.answer()


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()

    if text == BTN_ASK:
        await update.message.reply_text(
            "✍️ Напишите вопрос по документации, например:\n"
            "«Что такое ЭДО?» или «Для кого подходят бухгалтерские услуги?»",
            reply_markup=main_keyboard(),
        )
        return
    if text == BTN_STATS:
        await cmd_stats(update, context)
        return
    if text == BTN_EXAMPLES:
        await cmd_examples(update, context)
        return
    if text == BTN_CLEAR:
        await cmd_clear(update, context)
        return
    if text == BTN_HELP:
        await cmd_help(update, context)
        return

    await answer_question(update, context, text)


async def post_init(application: Application) -> None:
    await application.bot.set_my_commands(
        [
            BotCommand("start", "Главное меню"),
            BotCommand("help", "Справка"),
            BotCommand("examples", "Примеры вопросов"),
            BotCommand("stats", "Статистика"),
            BotCommand("clear", "Очистить кеш"),
        ]
    )
    logger.info("Команды меню Telegram установлены")


def build_pipeline() -> RAGPipeline:
    print("🚀 Инициализация RAG-системы для Telegram-бота...\n")
    pipeline = RAGPipeline(
        collection_name="api_rag_collection",
        cache_db_path="api_rag_cache.db",
        data_file="data/docs.txt",
        model="gpt-4o-mini",
    )
    print("\n✅ RAG-система готова.\n")
    return pipeline


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ Ошибка: переменная TELEGRAM_BOT_TOKEN не установлена")
        print("Добавьте её в assistant_api/.env:")
        print("  TELEGRAM_BOT_TOKEN=your-bot-token")
        sys.exit(1)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Ошибка: переменная OPENAI_API_KEY не установлена")
        sys.exit(1)

    pipeline = build_pipeline()
    application = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .build()
    )
    application.bot_data["pipeline"] = pipeline

    application.add_handler(CommandHandler("start", cmd_start))
    application.add_handler(CommandHandler("help", cmd_help))
    application.add_handler(CommandHandler("stats", cmd_stats))
    application.add_handler(CommandHandler("examples", cmd_examples))
    application.add_handler(CommandHandler("clear", cmd_clear))
    application.add_handler(CallbackQueryHandler(on_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    print("🤖 Telegram-бот запущен. Нажмите Ctrl+C для остановки.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
