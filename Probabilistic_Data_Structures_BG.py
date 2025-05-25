# Анализ на Bloom Filter

import numpy as np
import matplotlib.pyplot as plt
import hashlib
import time
from typing import List, Set
import mmh3  # За по-добри hash функции (може да се инсталира с pip install mmh3)

# Настройка на matplotlib за български текст
plt.rcParams['font.size'] = 12

class BloomFilter:
    """
    Реализация на Bloom Filter - вероятностна структура от данни
    за бързо тестване на членство в множество
    """
    
    def __init__(self, expected_items: int, false_positive_rate: float = 0.01):
        """
        Инициализира Bloom Filter
        
        Args:
            expected_items: Очакван брой елементи
            false_positive_rate: Желана вероятност за false positive (по подразбиране 1%)
        """
        self.expected_items = expected_items
        self.false_positive_rate = false_positive_rate
        
        # Изчисляване на оптималния размер на bit array
        self.size = self._optimal_size(expected_items, false_positive_rate)
        
        # Изчисляване на оптималния брой hash функции
        self.num_hashes = self._optimal_hash_count(self.size, expected_items)
        
        # Инициализиране на bit array
        self.bit_array = [0] * self.size
        
        # Брояч на добавени елементи
        self.item_count = 0
        
        print(f"Bloom Filter създаден:")
        print(f"  - Размер на bit array: {self.size}")
        print(f"  - Брой hash функции: {self.num_hashes}")
        print(f"  - Очаквани елементи: {expected_items}")
        print(f"  - Целева false positive rate: {false_positive_rate:.2%}")
    
    def _optimal_size(self, n: int, p: float) -> int:
        """
        Изчислява оптималния размер на bit array
        Formula: m = -(n * ln(p)) / (ln(2)^2)
        """
        import math
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)
    
    def _optimal_hash_count(self, m: int, n: int) -> int:
        """
        Изчислява оптималния брой hash функции
        Formula: k = (m/n) * ln(2)
        """
        import math
        k = (m / n) * math.log(2)
        return int(k)
    
    def _hash(self, item: str, seed: int) -> int:
        """
        Генерира hash стойност за даден елемент и seed
        """
        # Използваме различни методи за hash в зависимост от наличието на mmh3
        try:
            return mmh3.hash(item, seed) % self.size
        except:
            # Fallback към MD5 ако mmh3 не е наличен
            hash_obj = hashlib.md5((item + str(seed)).encode())
            return int(hash_obj.hexdigest(), 16) % self.size
    
    def add(self, item: str):
        """
        Добавя елемент към Bloom Filter
        """
        for i in range(self.num_hashes):
            index = self._hash(item, i)
            self.bit_array[index] = 1
        self.item_count += 1
    
    def might_contain(self, item: str) -> bool:
        """
        Проверява дали елементът МОЖЕ да е в множеството
        
        Returns:
            True: елементът МОЖЕ да е в множеството (или false positive)
            False: елементът ОПРЕДЕЛЕНО НЕ е в множеството
        """
        for i in range(self.num_hashes):
            index = self._hash(item, i)
            if self.bit_array[index] == 0:
                return False
        return True
    
    def current_false_positive_rate(self) -> float:
        """
        Изчислява текущата false positive rate
        """
        import math
        if self.item_count == 0:
            return 0.0
        
        # Formula: (1 - e^(-kn/m))^k
        k = self.num_hashes
        n = self.item_count
        m = self.size
        
        rate = (1 - math.exp(-k * n / m)) ** k
        return rate
    
    def get_statistics(self) -> dict:
        """
        Връща статистики за Bloom Filter
        """
        filled_bits = sum(self.bit_array)
        fill_ratio = filled_bits / self.size
        
        return {
            'size': self.size,
            'num_hashes': self.num_hashes,
            'item_count': self.item_count,
            'filled_bits': filled_bits,
            'fill_ratio': fill_ratio,
            'expected_fp_rate': self.false_positive_rate,
            'current_fp_rate': self.current_false_positive_rate()
        }

# Сравнение с традиционно множество
class TraditionalSet:
    """
    Традиционно множество за сравнение с Bloom Filter
    """
    
    def __init__(self):
        self.data = set()
    
    def add(self, item: str):
        self.data.add(item)
    
    def contains(self, item: str) -> bool:
        return item in self.data
    
    def size_in_bytes(self) -> int:
        """
        Приблизителен размер в байтове
        """
        return sum(len(item.encode('utf-8')) for item in self.data) + len(self.data) * 8

def demonstrate_bloom_filter():
    """
    Демонстрира работата на Bloom Filter
    """
    print("=" * 60)
    print("ДЕМОНСТРАЦИЯ НА BLOOM FILTER")
    print("=" * 60)
    
    # Създаване на Bloom Filter
    bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
    traditional_set = TraditionalSet()
    
    # Тестови данни
    test_data = [f"user_{i}@example.com" for i in range(500)]
    non_existent_data = [f"fake_{i}@test.com" for i in range(1000)]
    
    print(f"\nДобавяме {len(test_data)} елемента...")
    
    # Добавяне на данни
    start_time = time.time()
    for item in test_data:
        bf.add(item)
    bf_add_time = time.time() - start_time
    
    start_time = time.time()
    for item in test_data:
        traditional_set.add(item)
    set_add_time = time.time() - start_time
    
    # Тестване на съществуващи елементи
    print("\nТестване на съществуващи елементи:")
    existing_in_bf = sum(1 for item in test_data if bf.might_contain(item))
    existing_in_set = sum(1 for item in test_data if traditional_set.contains(item))
    
    print(f"Bloom Filter намери: {existing_in_bf}/{len(test_data)}")
    print(f"Traditional Set намери: {existing_in_set}/{len(test_data)}")
    
    # Тестване на несъществуващи елементи (false positives)
    print("\nТестване на несъществуващи елементи:")
    false_positives = sum(1 for item in non_existent_data if bf.might_contain(item))
    false_positives_set = sum(1 for item in non_existent_data if traditional_set.contains(item))
    
    actual_fp_rate = false_positives / len(non_existent_data)
    
    print(f"Bloom Filter false positives: {false_positives}/{len(non_existent_data)} ({actual_fp_rate:.2%})")
    print(f"Traditional Set false positives: {false_positives_set}/{len(non_existent_data)}")
    
    # Статистики
    stats = bf.get_statistics()
    print(f"\nСТАТИСТИКИ:")
    print(f"Размер на Bloom Filter: {stats['size']} bits ({stats['size'] // 8} bytes)")
    print(f"Размер на Traditional Set: {traditional_set.size_in_bytes()} bytes")
    print(f"Съотношение на размерите: {traditional_set.size_in_bytes() / (stats['size'] // 8):.1f}x")
    print(f"Запълненост на bit array: {stats['fill_ratio']:.2%}")
    print(f"Очаквана false positive rate: {stats['expected_fp_rate']:.2%}")
    print(f"Действителна false positive rate: {stats['current_fp_rate']:.2%}")
    
    print(f"\nВРЕМЕ ЗА ИЗПЪЛНЕНИЕ:")
    print(f"Bloom Filter добавяне: {bf_add_time:.4f}s")
    print(f"Traditional Set добавяне: {set_add_time:.4f}s")
    
    return bf, stats, actual_fp_rate

def analyze_false_positive_rates():
    """
    Анализира как false positive rate зависи от броя елементи
    """
    print("\n" + "=" * 60)
    print("АНАЛИЗ НА FALSE POSITIVE RATES")
    print("=" * 60)
    
    sizes = [100, 500, 1000, 2000, 5000]
    fp_rates = []
    theoretical_rates = []
    
    for size in sizes:
        bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
        
        # Добавяме различен брой елементи
        for i in range(size):
            bf.add(f"item_{i}")
        
        # Тестваме false positive rate
        test_items = [f"fake_{i}" for i in range(1000)]
        false_positives = sum(1 for item in test_items if bf.might_contain(item))
        actual_rate = false_positives / len(test_items)
        
        fp_rates.append(actual_rate)
        theoretical_rates.append(bf.current_false_positive_rate())
        
        print(f"Размер {size}: Действителна FP rate: {actual_rate:.3f}, "
              f"Теоретична: {bf.current_false_positive_rate():.3f}")
    
    # Графика
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, fp_rates, 'bo-', label='Действителна FP rate', linewidth=2)
    plt.plot(sizes, theoretical_rates, 'ro--', label='Теоретична FP rate', linewidth=2)
    plt.axhline(y=0.01, color='g', linestyle=':', label='Целева FP rate (1%)')
    
    plt.xlabel('Брой добавени елементи')
    plt.ylabel('False Positive Rate')
    plt.title('Зависимост на False Positive Rate от броя елементи')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def spell_checker_example():
    """
    Пример за употреба: Spell Checker
    """
    print("\n" + "=" * 60)
    print("ПРИМЕР: SPELL CHECKER С BLOOM FILTER")
    print("=" * 60)
    
    # Речник с български думи (опростен)
    bulgarian_words = [
        "автобус", "блокче", "време", "глава", "дума", "език", "жаба", "заек",
        "игра", "йогурт", "котка", "лампа", "мама", "нос", "око", "плод",
        "работа", "слон", "тарелка", "ухо", "филм", "хляб", "цвете", "чаша",
        "школо", "ябълка", "програма", "компютър", "интернет", "телефон",
        "книга", "учител", "студент", "университет", "математика", "физика",
        "химия", "биология", "история", "география", "литература", "изкуство"
    ]
    
    print(f"Създаваме речник с {len(bulgarian_words)} думи...")
    
    # Bloom Filter речник
    bf_dict = BloomFilter(expected_items=len(bulgarian_words), false_positive_rate=0.01)
    for word in bulgarian_words:
        bf_dict.add(word.lower())
    
    # Традиционен речник
    traditional_dict = set(word.lower() for word in bulgarian_words)
    
    # Тестови думи
    test_words = ["котка", "кученце", "програма", "асдфgh", "математика", "xyz123"]
    
    print(f"\nТестваме думи: {test_words}")
    print("\nРезултати:")
    print("-" * 50)
    
    for word in test_words:
        bf_result = bf_dict.might_contain(word.lower())
        traditional_result = word.lower() in traditional_dict
        
        status = "✓" if traditional_result else "✗"
        bf_status = "МОЖЕ ДА Е" if bf_result else "НЕ Е"
        
        print(f"{word:12} | Действителност: {status} | Bloom Filter: {bf_status}")
        
        if bf_result and not traditional_result:
            print(f"             | ⚠️  FALSE POSITIVE!")
    
    # Анализ на ефективността
    stats = bf_dict.get_statistics()
    traditional_size = sum(len(word.encode('utf-8')) for word in bulgarian_words)
    
    print(f"\nАНАЛИЗ НА ЕФЕКТИВНОСТТА:")
    print(f"Bloom Filter размер: {stats['size'] // 8} bytes")
    print(f"Традиционен речник: {traditional_size} bytes")
    print(f"Спестено място: {((traditional_size - stats['size'] // 8) / traditional_size) * 100:.1f}%")

def performance_comparison():
    """
    Сравнява производителността на различни структури
    """
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ НА ПРОИЗВОДИТЕЛНОСТТА")
    print("=" * 60)
    
    sizes = [1000, 5000, 10000, 50000]
    bf_times = []
    set_times = []
    
    # Увеличаваме броя на тестовете за по-точно измерване
    num_tests = 10000
    
    for size in sizes:
        print(f"\nТестване с {size} елемента...")
        
        # Подготовка на данни
        data = [f"item_{i}" for i in range(size)]
        test_data = [f"test_{i}" for i in range(num_tests)]
        
        # Bloom Filter
        bf = BloomFilter(expected_items=size, false_positive_rate=0.01)
        for item in data:
            bf.add(item)
        
        start_time = time.time()
        for item in test_data:
            bf.might_contain(item)
        bf_time = time.time() - start_time
        bf_times.append(bf_time)
        
        # Traditional Set
        traditional_set = set(data)
        
        start_time = time.time()
        for item in test_data:
            item in traditional_set
        set_time = time.time() - start_time
        set_times.append(set_time)
        
        print(f"Bloom Filter: {bf_time:.4f}s")
        print(f"Traditional Set: {set_time:.4f}s")
        
        # Избягваме деление на нула
        if bf_time > 0:
            ratio = set_time / bf_time
            print(f"Съотношение: {ratio:.2f}x")
        else:
            print("Съотношение: N/A (твърде бързо за измерване)")
    
    # Графика
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(sizes, bf_times, 'bo-', label='Bloom Filter', linewidth=2)
    plt.plot(sizes, set_times, 'ro-', label='Traditional Set', linewidth=2)
    plt.xlabel('Брой елементи в структурата')
    plt.ylabel(f'Време за {num_tests} проверки (секунди)')
    plt.title('Време за проверка на членство')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Избягваме деление на нула при изчисляване на ускорението
    speedup = []
    for i in range(len(sizes)):
        if bf_times[i] > 0:
            speedup.append(set_times[i] / bf_times[i])
        else:
            speedup.append(1.0)  # Ако времето е 0, приемаме че няма ускорение
    
    plt.plot(sizes, speedup, 'go-', linewidth=2)
    plt.xlabel('Брой елементи в структурата')
    plt.ylabel('Ускорение (пъти)')
    plt.title('Ускорение на Bloom Filter спрямо Set')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Теоретичен анализ
def theoretical_analysis():
    """
    Теоретичен анализ на Bloom Filter
    """
    print("\n" + "=" * 60)
    print("ТЕОРЕТИЧЕН АНАЛИЗ")
    print("=" * 60)
    
    print("""
    1. ЩО Е СТРУКТУРА ОТ ДАННИ?
    Структура от данни е начин за организиране и съхраняване на данни,
    който позволява ефективен достъп и модификация.
    
    2. ЩО Е ВЕРОЯТНОСТНА СТРУКТУРА ОТ ДАННИ?
    Вероятностна структура от данни използва рандомизация във своите операции
    или може да даде неточни резултати с известна вероятност.
    
    3. BLOOM FILTER - КОНСТРУКЦИЯ:
    - Bit array с m позиции (първоначално всички 0)
    - k независими hash функции
    - За добавяне на елемент: hash с всички k функции и постави 1 на съответните позиции
    - За проверка: hash с всички k функции и провери дали всички позиции са 1
    
    4. ОПЕРАЦИИ И СЛОЖНОСТ:
    - Добавяне: O(k) - константно време
    - Проверка за членство: O(k) - константно време
    - Изтриване: НЕ Е ВЪЗМОЖНО (деструктивно за други елементи)
    
    5. ВЕРОЯТНОСТИ:
    - False Positive: Възможен (структурата казва "може да е", а всъщност не е)
    - False Negative: НЕВЪЗМОЖЕН (ако структурата казва "не е", то определено не е)
    - Вероятност за False Positive: (1 - e^(-kn/m))^k
    
    6. ПРЕДИМСТВА:
    - Много малка памет (O(m) bits)
    - Бързи операции O(k)
    - Не съхранява действителните данни (privacy)
    
    7. НЕДОСТАТЪЦИ:
    - False positives
    - Невъзможност за изтриване
    - Няма достъп до действителните елементи
    
    8. УПОТРЕБА:
    - Spell checkers
    - Web crawlers (проверка за посетени URL-и)
    - Database systems (избягване на скъпи disk операции)
    - Cache systems
    - Network routers
    """)

def main():
    """
    Главна функция за изпълнение на всички анализи
    """
    print("АНАЛИЗ НА ВЕРОЯТНОСТНИ СТРУКТУРИ ОТ ДАННИ")
    print("Фокус: Bloom Filter")
    print("=" * 60)
    
    # Теоретичен анализ
    theoretical_analysis()
    
    # Демонстрация
    bf, stats, fp_rate = demonstrate_bloom_filter()
    
    # Анализ на false positive rates
    analyze_false_positive_rates()
    
    # Пример за употреба
    spell_checker_example()
    
    # Сравнение на производителността
    performance_comparison()
    
    print("\n" + "=" * 60)
    print("ЗАКЛЮЧЕНИЕ")
    print("=" * 60)
    print("""
    Bloom Filter е мощна вероятностна структура от данни, която:
    
    ✅ ПРЕДИМСТВА:
    - Изключително ефективна по памет (10-20x по-малко място)
    - Бързи операции (константно време)
    - Подходяща за системи с големи обеми данни
    - Гарантира липса на false negatives
    
    ⚠️ НЕДОСТАТЪЦИ:
    - Възможни false positives (контролируеми)
    - Невъзможност за изтриване на елементи
    - Не съхранява действителните данни
    
    🎯 УПОТРЕБА:
    Идеална за случаи където:
    - Паметта е ограничена
    - False positives са приемливи
    - Нужна е много бърза проверка за членство
    - Обемът данни е голям
    
    Примери: spell checkers, web crawlers, cache системи, мрежови рутери.
    """)

if __name__ == "__main__":
    main()
