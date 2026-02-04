"""
Basic Usage Example
Демонстрация базового использования Russian Accentor
"""

from accentor import load_accentor

def main():
    print("=" * 60)
    print("Russian Accentor - Basic Usage Example")
    print("=" * 60)
    
    # Загрузка модели
    print("\nЗагрузка модели...")
    accentor = load_accentor(
        model_path='model/acc_model.pt',
        vocab_path='model/vocab.json',
        device='auto'  # Автоопределение: cuda > mps > cpu
    )
    
    # Тестовые примеры
    test_sentences = [
        "Привет, как дела?",
        "Я иду домой через парк.",
        "Замок на замке был закрыт.",
        "Солнце светит ярко в небе.",
        "Это очень хороший день для прогулки.",
    ]
    
    print("\n" + "=" * 60)
    print("Примеры расстановки ударений")
    print("=" * 60)
    
    # Пример 1: Формат с апострофом
    print("\n1. ФОРМАТ С АПОСТРОФОМ (apostrophe)")
    print("-" * 60)
    for sentence in test_sentences:
        result = accentor(sentence, format='apostrophe')
        print(f"Вход:  {sentence}")
        print(f"Выход: {result}\n")
    
    # Пример 2: Формат для синтеза
    print("\n2. ФОРМАТ ДЛЯ СИНТЕЗА РЕЧИ (synthesis)")
    print("-" * 60)
    for sentence in test_sentences[:3]:  # Первые 3 для краткости
        result = accentor(sentence, format='synthesis')
        print(f"Вход:  {sentence}")
        print(f"Выход: {result}\n")
    
    # Пример 3: Оба формата сразу
    print("\n3. ОБА ФОРМАТА ОДНОВРЕМЕННО (both)")
    print("-" * 60)
    text = "Мама мыла раму в красивой раме."
    apostrophe, synthesis = accentor(text, format='both')
    print(f"Вход:      {text}")
    print(f"Апостроф:  {apostrophe}")
    print(f"Синтез:    {synthesis}")
    
    # Пример 4: Обработка списка
    print("\n4. ПАКЕТНАЯ ОБРАБОТКА")
    print("-" * 60)
    batch_texts = [
        "Первое предложение.",
        "Второе предложение.",
        "Третье предложение."
    ]
    results = accentor(batch_texts, format='apostrophe')
    for original, accented in zip(batch_texts, results):
        print(f"{original:30} → {accented}")
    
    # Информация о кэше
    print("\n" + "=" * 60)
    print("Информация о кэше")
    print("=" * 60)
    cache_info = accentor.cache_info()
    print(f"Размер кэша: {cache_info['size']} записей")
    print(f"Попадания:   {cache_info['hits']}")
    print(f"Промахи:     {cache_info['misses']}")
    if cache_info['hits'] + cache_info['misses'] > 0:
        hit_rate = cache_info['hits'] / (cache_info['hits'] + cache_info['misses'])
        print(f"Процент попаданий: {hit_rate:.1%}")
    
    print("\n" + "=" * 60)
    print("Пример завершён!")
    print("=" * 60)


if __name__ == "__main__":
    main()
