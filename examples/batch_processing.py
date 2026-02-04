"""
Batch Processing Example
Демонстрация пакетной обработки больших объемов текста
"""

import time
from pathlib import Path
from accentor import load_accentor


def process_file(input_path: str, output_path: str, batch_size: int = 32, 
                 format: str = 'apostrophe'):
    """
    Обрабатывает текстовый файл и сохраняет результаты
    
    Args:
        input_path: Путь к входному файлу
        output_path: Путь к выходному файлу
        batch_size: Размер батча для обработки
        format: Формат вывода ('apostrophe' или 'synthesis')
    """
    print("=" * 70)
    print("Batch Processing Example - Обработка файла")
    print("=" * 70)
    
    # Загрузка модели
    print("\n📖 Загрузка модели...")
    start_load = time.time()
    accentor = load_accentor(
        model_path='model/acc_model.pt',
        vocab_path='model/vocab.json',
        device='auto'
    )
    load_time = time.time() - start_load
    print(f"✅ Модель загружена за {load_time:.2f} сек")
    
    # Чтение входного файла
    print(f"\n📄 Чтение файла: {input_path}")
    input_file = Path(input_path)
    
    if not input_file.exists():
        print(f"❌ Файл не найден: {input_path}")
        print("💡 Создайте файл input.txt с примерами текста")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"📊 Загружено строк: {len(lines)}")
    
    if not lines:
        print("⚠️  Файл пуст!")
        return
    
    # Обработка
    print(f"\n⚡ Обработка с batch_size={batch_size}...")
    start_process = time.time()
    
    results = accentor(lines, format=format, batch_size=batch_size)
    
    process_time = time.time() - start_process
    
    # Статистика
    speed = len(lines) / process_time if process_time > 0 else 0
    print(f"\n📈 Статистика обработки:")
    print(f"   Обработано строк: {len(lines)}")
    print(f"   Время обработки: {process_time:.2f} сек")
    print(f"   Скорость: {speed:.1f} строк/сек")
    print(f"   Среднее время на строку: {process_time/len(lines)*1000:.1f} мс")
    
    # Информация о кэше
    cache_info = accentor.cache_info()
    print(f"\n💾 Кэш:")
    print(f"   Размер: {cache_info['size']} записей")
    print(f"   Попадания: {cache_info['hits']}")
    print(f"   Промахи: {cache_info['misses']}")
    
    # Сохранение результатов
    print(f"\n💾 Сохранение результатов: {output_path}")
    output_file = Path(output_path)
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')
    
    print(f"✅ Результаты сохранены!")
    
    # Показываем несколько примеров
    print(f"\n📝 Примеры результатов (первые 5):")
    print("-" * 70)
    for i, (original, accented) in enumerate(zip(lines[:5], results[:5]), 1):
        print(f"{i}. Вход:  {original}")
        print(f"   Выход: {accented}\n")
    
    print("=" * 70)
    print("✅ Обработка завершена!")
    print("=" * 70)


def create_sample_input():
    """Создает пример входного файла если его нет"""
    sample_file = Path("input.txt")
    
    if sample_file.exists():
        print(f"✅ Файл {sample_file} уже существует")
        return
    
    print(f"📝 Создание примера входного файла: {sample_file}")
    
    sample_texts = [
        "Привет, как дела?",
        "Я иду домой через парк.",
        "Солнце светит ярко в небе.",
        "Это очень хороший день для прогулки.",
        "Дети играют во дворе с мячом.",
        "Кошка спит на теплом подоконнике.",
        "Мама готовит вкусный обед на кухне.",
        "Папа читает интересную книгу в кресле.",
        "Бабушка вяжет красивый шарф из шерсти.",
        "Дедушка работает в саду с лопатой.",
        "Птицы поют песни на высоких деревьях.",
        "Машины едут по широкой дороге в город.",
        "Студенты учатся в большой библиотеке.",
        "Художник рисует картину красивого пейзажа.",
        "Музыкант играет на скрипке старинную мелодию.",
    ]
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        for text in sample_texts:
            f.write(text + '\n')
    
    print(f"✅ Создано {len(sample_texts)} примеров в {sample_file}")


def compare_formats():
    """Сравнивает оба формата вывода"""
    print("=" * 70)
    print("Сравнение форматов вывода")
    print("=" * 70)
    
    accentor = load_accentor(
        model_path='model/acc_model.pt',
        vocab_path='model/vocab.json'
    )
    
    test_texts = [
        "Замок на замке был закрыт на замок.",
        "Мама мыла раму в красивой раме.",
        "Я иду домой через темный лес.",
    ]
    
    print("\nСравнение форматов:\n")
    
    for i, text in enumerate(test_texts, 1):
        apostrophe, synthesis = accentor(text, format='both')
        
        print(f"{i}. Исходный текст:")
        print(f"   {text}")
        print(f"\n   Формат Apostrophe (апостроф после гласной):")
        print(f"   {apostrophe}")
        print(f"\n   Формат Synthesis (+ перед гласной):")
        print(f"   {synthesis}")
        print("\n" + "-" * 70 + "\n")


def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch processing example for Russian Accentor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Создать пример входного файла
  python example_batch.py --create-sample

  # Обработать файл с форматом апостроф
  python example_batch.py --input input.txt --output output.txt

  # Обработать с форматом для синтеза
  python example_batch.py --input input.txt --output output.txt --format synthesis

  # Обработать с большим batch size для скорости
  python example_batch.py --input input.txt --batch-size 64

  # Сравнить оба формата
  python example_batch.py --compare-formats
        """
    )
    
    parser.add_argument('--input', default='input.txt',
                       help='Входной текстовый файл')
    parser.add_argument('--output', default='output.txt',
                       help='Выходной текстовый файл')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Размер батча (по умолчанию: 32)')
    parser.add_argument('--format', choices=['apostrophe', 'synthesis'],
                       default='apostrophe',
                       help='Формат вывода')
    parser.add_argument('--create-sample', action='store_true',
                       help='Создать пример входного файла')
    parser.add_argument('--compare-formats', action='store_true',
                       help='Сравнить оба формата вывода')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_input()
    elif args.compare_formats:
        compare_formats()
    else:
        process_file(
            input_path=args.input,
            output_path=args.output,
            batch_size=args.batch_size,
            format=args.format
        )


if __name__ == "__main__":
    main()
