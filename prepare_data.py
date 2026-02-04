"""
01_prepare_data.py
Извлечение пар предложений (без ударений → с ударениями) в CSV

Шаг 1: Извлекаем ВСЕ валидные пары в CSV
Шаг 2: Создаем детальную статистику с анализом словаря
"""

import re
import csv
import json
from pathlib import Path
from typing import List, Tuple, Dict, Set
from collections import Counter
from tqdm import tqdm


class SentencePairExtractor:
    """Извлекает пары предложений из книг с ударениями"""
    
    def __init__(self):
        self.stats = {
            'total_lines': 0,
            'valid_pairs': 0,
            'discarded_no_stress': 0,
            'discarded_headers': 0,
            'discarded_too_short': 0,
            'discarded_too_long': 0,
            'discarded_no_ending': 0,
        }
        
        # Для статистики
        self.all_words = []  # Все слова (для частотного словаря)
        self.all_lemmas = set()  # Уникальные основы слов (упрощенная версия)
        
    def remove_stress_marks(self, text: str) -> str:
        """
        Убирает только ударения (апострофы), сохраняя букву ё
        
        Args:
            text: текст с ударениями "Вы'дался тёплый"
        Returns:
            текст без ударений: "Выдался тёплый"
        """
        # Убираем только апострофы (включая оба варианта)
        text = text.replace("'", "").replace("'", "")
        return text
    
    def is_valid_sentence(self, text: str) -> Tuple[bool, str]:
        """
        Проверяет, является ли строка полным предложением
        
        Returns:
            (is_valid, reason)
        """
        text = text.strip()
        
        # Минимальная длина
        if len(text) < 20:
            return False, 'too_short'
        
        # Максимальная длина
        if len(text) > 512:
            return False, 'too_long'
        
        # Должно начинаться с заглавной буквы
        if not text[0].isupper():
            return False, 'no_capital'
        
        # Должно заканчиваться на . ! ? (знак препинания)
        if not text[-1] in '.!?':
            return False, 'no_ending'
        
        # Проверяем количество слов
        # Заголовки обычно короткие: "Часть I", "Глава II"
        words = text.split()
        if len(words) < 5:
            return False, 'header'
        
        # Проверяем наличие ударений
        if "'" not in text and "'" not in text:
            return False, 'no_stress'
        
        return True, 'ok'
    
    def extract_words(self, text: str) -> List[str]:
        """
        Извлекает слова из текста (только русские буквы)
        """
        # Убираем апострофы для анализа
        text_clean = self.remove_stress_marks(text)
        
        # Извлекаем только слова (русские буквы + ё)
        words = re.findall(r'[а-яёА-ЯЁ]+', text_clean)
        return [w.lower() for w in words]
    
    def simple_lemmatize(self, word: str) -> str:
        """
        Простая лемматизация - убирает распространенные окончания
        Не идеально, но работает для статистики
        
        Args:
            word: слово в любой форме
        Returns:
            упрощенная основа слова
        """
        word = word.lower()
        
        # Убираем распространенные окончания
        endings = [
            'ами', 'ями', 'ов', 'ев', 'ам', 'ям', 'ах', 'ях',
            'ом', 'ем', 'ой', 'ей', 'ою', 'ею', 'ого', 'его',
            'ому', 'ему', 'ым', 'им', 'ую', 'юю', 'ая', 'яя',
            'ое', 'ее', 'ие', 'ые', 'их', 'ых', 'ими', 'ыми',
            'а', 'я', 'у', 'ю', 'ы', 'и', 'е', 'о'
        ]
        
        # Сортируем по длине (сначала длинные)
        endings.sort(key=len, reverse=True)
        
        for ending in endings:
            if word.endswith(ending) and len(word) > len(ending) + 2:
                return word[:-len(ending)]
        
        return word
    
    def process_book_file(self, file_path: Path) -> List[Tuple[str, str]]:
        """
        Обрабатывает один файл книги
        
        Returns:
            список пар (source, target)
        """
        pairs = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in tqdm(lines, desc=f"📖 {file_path.name}"):
            self.stats['total_lines'] += 1
            
            line = line.strip()
            if not line:
                continue
            
            # Проверяем валидность
            is_valid, reason = self.is_valid_sentence(line)
            
            if not is_valid:
                # Подсчитываем причины отбрасывания
                if reason == 'no_stress':
                    self.stats['discarded_no_stress'] += 1
                elif reason == 'header':
                    self.stats['discarded_headers'] += 1
                elif reason == 'too_short':
                    self.stats['discarded_too_short'] += 1
                elif reason == 'too_long':
                    self.stats['discarded_too_long'] += 1
                elif reason == 'no_ending':
                    self.stats['discarded_no_ending'] += 1
                continue
            
            # Создаем пару
            target = line  # С ударениями
            source = self.remove_stress_marks(target)  # Без ударений
            
            pairs.append((source, target))
            self.stats['valid_pairs'] += 1
            
            # Собираем слова для статистики
            words = self.extract_words(target)
            self.all_words.extend(words)
            
            # Собираем леммы (упрощенные основы)
            for word in words:
                lemma = self.simple_lemmatize(word)
                self.all_lemmas.add(lemma)
        
        return pairs
    
    def save_to_csv(self, pairs: List[Tuple[str, str]], output_file: Path):
        """
        Сохраняет пары в CSV файл
        """
        print(f"\n💾 Сохранение в CSV: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            
            # Заголовок
            writer.writerow(['source', 'target', 'length', 'stress_count', 'word_count'])
            
            # Данные
            for source, target in tqdm(pairs, desc="Запись"):
                length = len(target)
                stress_count = target.count("'") + target.count("'")
                word_count = len(target.split())
                
                writer.writerow([source, target, length, stress_count, word_count])
        
        print(f"✅ Сохранено {len(pairs)} пар")
    
    def create_statistics(self, pairs: List[Tuple[str, str]], output_dir: Path):
        """
        Создает детальную статистику
        """
        print("\n📊 Создание статистики...")
        
        # Базовая статистика
        lengths = [len(target) for _, target in pairs]
        stress_counts = [target.count("'") + target.count("'") for _, target in pairs]
        word_counts = [len(target.split()) for _, target in pairs]
        
        # Частотный словарь
        word_freq = Counter(self.all_words)
        top_words = word_freq.most_common(100000)  # Топ 100k слов
        
        # Статистика
        stats = {
            'extraction_stats': {
                'total_lines_processed': self.stats['total_lines'],
                'valid_pairs_extracted': self.stats['valid_pairs'],
                'discarded_total': sum([
                    self.stats['discarded_no_stress'],
                    self.stats['discarded_headers'],
                    self.stats['discarded_too_short'],
                    self.stats['discarded_too_long'],
                    self.stats['discarded_no_ending'],
                ]),
                'discarded_breakdown': {
                    'no_stress': self.stats['discarded_no_stress'],
                    'headers': self.stats['discarded_headers'],
                    'too_short': self.stats['discarded_too_short'],
                    'too_long': self.stats['discarded_too_long'],
                    'no_ending': self.stats['discarded_no_ending'],
                }
            },
            
            'sentence_stats': {
                'total_pairs': len(pairs),
                'avg_length': sum(lengths) / len(lengths) if lengths else 0,
                'min_length': min(lengths) if lengths else 0,
                'max_length': max(lengths) if lengths else 0,
                'avg_stress_marks': sum(stress_counts) / len(stress_counts) if stress_counts else 0,
                'avg_words_per_sentence': sum(word_counts) / len(word_counts) if word_counts else 0,
            },
            
            'vocabulary_stats': {
                'total_words': len(self.all_words),
                'unique_words_inflected': len(set(self.all_words)),
                'unique_words_lemmas_approx': len(self.all_lemmas),
                'top_100_words': word_freq.most_common(100),
            },
            
            'coverage': {
                'top_1000_coverage': sum([count for word, count in top_words[:1000]]) / len(self.all_words) * 100 if self.all_words else 0,
                'top_5000_coverage': sum([count for word, count in top_words[:5000]]) / len(self.all_words) * 100 if self.all_words else 0,
                'top_10000_coverage': sum([count for word, count in top_words[:10000]]) / len(self.all_words) * 100 if self.all_words else 0,
            }
        }
        
        # Сохраняем основную статистику
        with open(output_dir / 'statistics.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Статистика сохранена: statistics.json")
        
        # Сохраняем частотный словарь (топ 100k)
        with open(output_dir / 'word_frequency_100k.json', 'w', encoding='utf-8') as f:
            freq_dict = {word: count for word, count in top_words}
            json.dump(freq_dict, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Частотный словарь сохранен: word_frequency_100k.json")
        
        # Сохраняем список уникальных основ слов
        with open(output_dir / 'unique_lemmas.txt', 'w', encoding='utf-8') as f:
            for lemma in sorted(self.all_lemmas):
                f.write(lemma + '\n')
        
        print(f"✅ Уникальные основы слов (приблизительно): unique_lemmas.txt ({len(self.all_lemmas)} слов)")
        
        # Печатаем краткую статистику
        self._print_summary(stats)
    
    def _print_summary(self, stats: Dict):
        """Печатает краткую сводку"""
        print("\n" + "=" * 70)
        print("📈 СТАТИСТИКА ИЗВЛЕЧЕНИЯ")
        print("=" * 70)
        
        print(f"\n📚 Обработка:")
        print(f"   Всего строк обработано: {stats['extraction_stats']['total_lines_processed']:,}")
        print(f"   ✅ Валидных пар: {stats['extraction_stats']['valid_pairs_extracted']:,}")
        print(f"   ❌ Отброшено: {stats['extraction_stats']['discarded_total']:,}")
        
        print(f"\n📊 Разбивка отброшенных:")
        for reason, count in stats['extraction_stats']['discarded_breakdown'].items():
            print(f"   - {reason}: {count:,}")
        
        print(f"\n📏 Предложения:")
        print(f"   Средняя длина: {stats['sentence_stats']['avg_length']:.1f} символов")
        print(f"   Диапазон: {stats['sentence_stats']['min_length']} - {stats['sentence_stats']['max_length']}")
        print(f"   Среднее ударений: {stats['sentence_stats']['avg_stress_marks']:.1f}")
        print(f"   Среднее слов: {stats['sentence_stats']['avg_words_per_sentence']:.1f}")
        
        print(f"\n📖 Словарь:")
        print(f"   Всего словоупотреблений: {stats['vocabulary_stats']['total_words']:,}")
        print(f"   Уникальных слов (с окончаниями): {stats['vocabulary_stats']['unique_words_inflected']:,}")
        print(f"   Уникальных слов (основы, ~): {stats['vocabulary_stats']['unique_words_lemmas_approx']:,}")
        
        print(f"\n🎯 Покрытие частотным словарем:")
        print(f"   Топ-1000 слов: {stats['coverage']['top_1000_coverage']:.1f}%")
        print(f"   Топ-5000 слов: {stats['coverage']['top_5000_coverage']:.1f}%")
        print(f"   Топ-10000 слов: {stats['coverage']['top_10000_coverage']:.1f}%")
        
        print(f"\n🔤 Топ-20 самых частых слов:")
        for i, (word, count) in enumerate(stats['vocabulary_stats']['top_100_words'][:20], 1):
            print(f"   {i:2d}. {word:15s} - {count:,} раз")


def main():
    """
    Основная функция
    """
    print("=" * 70)
    print("📚 ИЗВЛЕЧЕНИЕ ПАР ПРЕДЛОЖЕНИЙ ДЛЯ ОБУЧЕНИЯ МОДЕЛИ УДАРЕНИЙ")
    print("=" * 70)
    print("\nШаг 1: Извлечение всех валидных пар в CSV")
    print("Шаг 2: Создание детальной статистики\n")
    
    # Настройки
    BOOKS_DIR = Path("./books")
    OUTPUT_DIR = Path("./temp")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Проверяем наличие книг
    if not BOOKS_DIR.exists():
        print(f"❌ Ошибка: папка {BOOKS_DIR} не найдена!")
        print(f"   Создайте папку './books' и положите туда .txt файлы")
        return
    
    book_files = list(BOOKS_DIR.glob("*.txt"))
    if not book_files:
        print(f"❌ Ошибка: в {BOOKS_DIR} нет .txt файлов!")
        return
    
    print(f"📖 Найдено книг: {len(book_files)}")
    for i, book in enumerate(book_files[:10], 1):
        print(f"   {i:2d}. {book.name}")
    if len(book_files) > 10:
        print(f"   ... и еще {len(book_files) - 10} книг")
    
    # Создаем экстрактор
    print("\n" + "-" * 70)
    extractor = SentencePairExtractor()
    
    # Обрабатываем все книги
    all_pairs = []
    
    for book_file in book_files:
        pairs = extractor.process_book_file(book_file)
        all_pairs.extend(pairs)
    
    print(f"\n✅ Обработка завершена!")
    print(f"   Извлечено пар: {len(all_pairs):,}")
    
    # Сохраняем в CSV
    csv_file = OUTPUT_DIR / "sentence_pairs.csv"
    extractor.save_to_csv(all_pairs, csv_file)
    
    # Создаем статистику
    extractor.create_statistics(all_pairs, OUTPUT_DIR)
    
    # Сохраняем примеры
    samples_file = OUTPUT_DIR / "samples.txt"
    print(f"\n📝 Сохранение примеров: {samples_file}")
    with open(samples_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("ПРИМЕРЫ ИЗВЛЕЧЕННЫХ ПАР\n")
        f.write("=" * 70 + "\n\n")
        
        for i, (source, target) in enumerate(all_pairs[:50], 1):
            stress_count = target.count("'") + target.count("'")
            f.write(f"Пример {i}:\n")
            f.write(f"  Вход:  {source}\n")
            f.write(f"  Выход: {target}\n")
            f.write(f"  Длина: {len(target)} символов, ударений: {stress_count}\n\n")
    
    print(f"✅ Примеры сохранены (первые 50)")
    
    print("\n" + "=" * 70)
    print("✅ ГОТОВО!")
    print("=" * 70)
    print(f"\n📁 Результаты в папке: {OUTPUT_DIR}/")
    print(f"   📄 sentence_pairs.csv - все пары предложений")
    print(f"   📊 statistics.json - детальная статистика")
    print(f"   📖 word_frequency_100k.json - частотный словарь (100k слов)")
    print(f"   📝 unique_lemmas.txt - уникальные основы слов (приблизительно)")
    print(f"   📝 samples.txt - примеры (первые 50 пар)")
    print("\n🎯 Следующий шаг: анализ и фильтрация для разнообразия\n")


if __name__ == "__main__":
    main()