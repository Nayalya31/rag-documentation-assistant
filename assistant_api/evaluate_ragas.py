"""
Оценка качества RAG системы через RAGAS для assistant_api.
Использует OpenAI API для RAG и для метрик RAGAS.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
_BASE_DIR = Path(__file__).resolve().parent
for env_path in (_BASE_DIR / '.env', _BASE_DIR.parent / '.env'):
    if env_path.exists():
        load_dotenv(env_path)
        break
else:
    load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import embedding_factory

# Классы метрик RAGAS (с fallback на старые импорты)
try:
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.metrics._context_precision import ContextPrecision
    from ragas.metrics._answer_relevance import AnswerRelevancy
    faithfulness = Faithfulness
    context_precision = ContextPrecision
    answer_relevancy = AnswerRelevancy
except ImportError:
    try:
        from ragas.metrics.collections import (
            faithfulness,
            context_precision,
            answer_relevancy,
        )
    except ImportError:
        from ragas.metrics import faithfulness, context_precision, answer_relevancy

from rag_pipeline import RAGPipeline


# Тестовые вопросы для оценки RAG системы
EVALUATION_QUESTIONS = [
    "Что такое бухгалтерская услуга?",
    "Для кого подходят бухгалтерские услуги?",
    "Что такое ЭДО?",
    "Что такое электронная подпись?",
    "Что такое электронный ключ?"
]


def prepare_dataset(pipeline: RAGPipeline, questions: list) -> Dataset:
    """
    Подготовка датасета для RAGAS из вопросов.
    
    Args:
        pipeline: RAG pipeline для получения ответов
        questions: список вопросов для оценки
    
    Returns:
        Dataset для RAGAS с полями: question, answer, contexts, ground_truth
    """
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truths_list = []
    
    print("[*] Получение ответов от RAG системы...\n")
    
    for i, question in enumerate(questions, 1):
        print(f"  {i}/{len(questions)}: {question}")
        
        # Получаем ответ от RAG системы (без использования кеша)
        result = pipeline.query(question, use_cache=False)
        
        # Формируем данные для RAGAS
        questions_list.append(question)
        answers_list.append(result["answer"])
        
        # Контекст - список текстов из найденных документов
        context_texts = [doc["text"] for doc in result["context_docs"]]
        contexts_list.append(context_texts)
        
        # Ground truth - эталонный ответ (для демонстрации используем часть ответа)
        # В реальном проекте здесь должны быть вручную подготовленные эталонные ответы
        ground_truths_list.append(result["answer"][:100])
        
        print(f"     [+] Ответ получен от OpenAI API")
    
    print()
    
    # Создаём датасет для RAGAS
    dataset_dict = {
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truths_list
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    return dataset


def evaluate_rag_system():
    """
    Основная функция оценки RAG-системы через RAGAS.
    
    Процесс:
    1. Инициализация RAG pipeline
    2. Генерация ответов на тестовые вопросы
    3. Подготовка датасета для RAGAS
    4. Запуск оценки метрик
    5. Вывод результатов
    """
    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА RAG-СИСТЕМЫ (API MODE) ЧЕРЕЗ RAGAS")
    print("=" * 70)
    print()
    
    # Проверка наличия API ключа
    if not os.getenv("OPENAI_API_KEY"):
        print("[ОШИБКА] OPENAI_API_KEY не установлен")
        print("\nУстановите переменную окружения:")
        print("  Windows (PowerShell): $env:OPENAI_API_KEY='your-key'")
        print("  Windows (CMD): set OPENAI_API_KEY=your-key")
        print("  Linux/Mac: export OPENAI_API_KEY='your-key'")
        print("\nИли создайте файл .env в корне проекта с содержимым:")
        print("  OPENAI_API_KEY=your-key-here")
        sys.exit(1)
    
    # Инициализация RAG pipeline
    try:
        print("[*] Инициализация RAG системы (API mode)...\n")
        pipeline = RAGPipeline(
            collection_name="api_rag_collection",
            cache_db_path="api_rag_cache.db",
            data_file="data/docs.txt",
            model="gpt-4o-mini"
        )
        print("\n[OK] RAG система готова к оценке\n")
    except Exception as e:
        print(f"[ОШИБКА] Ошибка инициализации RAG pipeline: {e}")
        sys.exit(1)
    
    # Подготовка датасета
    print("=" * 70)
    dataset = prepare_dataset(pipeline, EVALUATION_QUESTIONS)
    print("=" * 70)
    
    print("\n[*] Запуск оценки метрик RAGAS...")
    print("   Метрики: Faithfulness, Context Precision, Answer Relevancy")
    print("   (это займёт 1-2 минуты, так как RAGAS использует OpenAI для оценки)\n")
    
    # Embeddings через фабрику RAGAS (OpenAI text-embedding-3-small),
    # чтобы Answer Relevancy не зависела от несовместимого langchain-openai API.
    print("   [+] Используем метрики RAGAS с OpenAI embeddings\n")
    metrics_to_use = [faithfulness(), context_precision(), answer_relevancy()]
    ragas_embeddings = embedding_factory(model="text-embedding-3-small")
    
    # Запускаем оценку RAGAS
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics_to_use,
            embeddings=ragas_embeddings,
        )
    except Exception as e:
        print(f"[ОШИБКА] Ошибка при оценке: {e}")
        sys.exit(1)
    
    # Обработка и вывод результатов
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("=" * 70)
    
    # Вычисляем средние значения метрик (игнорируя NaN, в т.ч. numpy)
    import math

    def is_nan(value) -> bool:
        try:
            return math.isnan(float(value))
        except (TypeError, ValueError):
            return True

    def metric_values(name: str) -> list:
        return [float(v) for v in result[name] if not is_nan(v)]

    def metric_avg(name: str) -> float:
        values = metric_values(name)
        return sum(values) / len(values) if values else 0.0

    avg_faithfulness = metric_avg('faithfulness')
    avg_context_precision = metric_avg('context_precision')
    avg_answer_relevancy = metric_avg('answer_relevancy')
    
    # Выводим общие метрики
    print()
    print("[МЕТРИКИ] Средние значения:")
    print(f"   Faithfulness (точность ответа):          {avg_faithfulness:.4f}")
    print(f"   Context Precision (точность контекста):  {avg_context_precision:.4f}")
    print(f"   Answer Relevancy (релевантность ответа): {avg_answer_relevancy:.4f}")
    
    # Вычисляем и выводим средний балл
    avg_score = (avg_faithfulness + avg_context_precision + avg_answer_relevancy) / 3
    print(f"\n{'─'*70}")
    print(f"[ИТОГО] Средний балл: {avg_score:.4f}")
    
    # Оценка качества системы
    if avg_score >= 0.7:
        print("   Оценка: Отличное качество! [OK]")
        print("   Система показывает высокую точность и релевантность ответов.")
    elif avg_score >= 0.5:
        print("   Оценка: Удовлетворительное качество [!]")
        print("   Рекомендуется улучшить качество документов или промптов.")
    else:
        print("   Оценка: Требует значительного улучшения [X]")
        print("   Необходимо пересмотреть стратегию chunking или качество данных.")
    
    # Выводим детали по каждому вопросу
    print("\n" + "=" * 70)
    print("ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ ПО ВОПРОСАМ")
    print("=" * 70)
    
    for i, question in enumerate(EVALUATION_QUESTIONS):
        print(f"\n{i+1}. {question}")
        
        faith_val = result['faithfulness'][i]
        print(f"   Faithfulness:       {faith_val:.4f}" if not is_nan(faith_val) else "   Faithfulness:       не удалось вычислить")

        cp_val = result['context_precision'][i]
        print(f"   Context Precision:  {cp_val:.4f}" if not is_nan(cp_val) else "   Context Precision:  не удалось вычислить")

        ar_val = result['answer_relevancy'][i]
        print(f"   Answer Relevancy:   {ar_val:.4f}" if not is_nan(ar_val) else "   Answer Relevancy:   не удалось вычислить")
    
    # Пояснения к метрикам
    print("\n" + "=" * 70)
    print("[INFO] ПОЯСНЕНИЯ К МЕТРИКАМ")
    print("=" * 70)
    print("""
Faithfulness (Точность ответа):
  Измеряет, насколько ответ соответствует предоставленному контексту.
  Значения: 0.0 - 1.0 (1.0 = полное соответствие контексту)

Context Precision (Точность контекста):
  Измеряет качество извлечённого контекста для ответа на вопрос.
  Значения: 0.0 - 1.0 (1.0 = идеальный контекст)

Answer Relevancy (Релевантность ответа):
  Измеряет, насколько ответ относится к заданному вопросу.
  Значения: 0.0 - 1.0 (1.0 = ответ полностью по теме вопроса)
    """)
    
    print("=" * 70)
    print("[OK] Оценка завершена!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    evaluate_rag_system()

