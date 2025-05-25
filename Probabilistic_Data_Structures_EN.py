# Bloom Filter Analysis

import numpy as np
import matplotlib.pyplot as plt
import hashlib
import time
from typing import List, Set
import mmh3  # For better hash functions (can be installed with pip install mmh3)

# Set matplotlib font size
plt.rcParams['font.size'] = 12

class BloomFilter:
    """
    Implementation of Bloom Filter - a probabilistic data structure
    for fast membership testing in sets
    """
    
    def __init__(self, expected_items: int, false_positive_rate: float = 0.01):
        """
        Initialize Bloom Filter
        
        Args:
            expected_items: Expected number of items
            false_positive_rate: Desired false positive rate (default 1%)
        """
        self.expected_items = expected_items
        self.false_positive_rate = false_positive_rate
        
        # Calculate optimal size of bit array
        self.size = self._optimal_size(expected_items, false_positive_rate)
        
        # Calculate optimal number of hash functions
        self.num_hashes = self._optimal_hash_count(self.size, expected_items)
        
        # Initialize bit array
        self.bit_array = [0] * self.size
        
        # Counter for added items
        self.item_count = 0
        
        print(f"Bloom Filter created:")
        print(f"  - Bit array size: {self.size}")
        print(f"  - Number of hash functions: {self.num_hashes}")
        print(f"  - Expected items: {expected_items}")
        print(f"  - Target false positive rate: {false_positive_rate:.2%}")
    
    def _optimal_size(self, n: int, p: float) -> int:
        """
        Calculate optimal size of bit array
        Formula: m = -(n * ln(p)) / (ln(2)^2)
        """
        import math
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)
    
    def _optimal_hash_count(self, m: int, n: int) -> int:
        """
        Calculate optimal number of hash functions
        Formula: k = (m/n) * ln(2)
        """
        import math
        k = (m / n) * math.log(2)
        return int(k)
    
    def _hash(self, item: str, seed: int) -> int:
        """
        Generate hash value for given item and seed
        """
        # Use different hash methods depending on mmh3 availability
        try:
            return mmh3.hash(item, seed) % self.size
        except:
            # Fallback to MD5 if mmh3 is not available
            hash_obj = hashlib.md5((item + str(seed)).encode())
            return int(hash_obj.hexdigest(), 16) % self.size
    
    def add(self, item: str):
        """
        Add element to Bloom Filter
        """
        for i in range(self.num_hashes):
            index = self._hash(item, i)
            self.bit_array[index] = 1
        self.item_count += 1
    
    def might_contain(self, item: str) -> bool:
        """
        Check if element MIGHT be in the set
        
        Returns:
            True: element MIGHT be in the set (or false positive)
            False: element is DEFINITELY NOT in the set
        """
        for i in range(self.num_hashes):
            index = self._hash(item, i)
            if self.bit_array[index] == 0:
                return False
        return True
    
    def current_false_positive_rate(self) -> float:
        """
        Calculate current false positive rate
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
        Return statistics for Bloom Filter
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

# Comparison with traditional set
class TraditionalSet:
    """
    Traditional set for comparison with Bloom Filter
    """
    
    def __init__(self):
        self.data = set()
    
    def add(self, item: str):
        self.data.add(item)
    
    def contains(self, item: str) -> bool:
        return item in self.data
    
    def size_in_bytes(self) -> int:
        """
        Approximate size in bytes
        """
        return sum(len(item.encode('utf-8')) for item in self.data) + len(self.data) * 8

def demonstrate_bloom_filter():
    """
    Demonstrate Bloom Filter functionality
    """
    print("=" * 60)
    print("BLOOM FILTER DEMONSTRATION")
    print("=" * 60)
    
    # Create Bloom Filter
    bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
    traditional_set = TraditionalSet()
    
    # Test data
    test_data = [f"user_{i}@example.com" for i in range(500)]
    non_existent_data = [f"fake_{i}@test.com" for i in range(1000)]
    
    print(f"\nAdding {len(test_data)} items...")
    
    # Add data
    start_time = time.time()
    for item in test_data:
        bf.add(item)
    bf_add_time = time.time() - start_time
    
    start_time = time.time()
    for item in test_data:
        traditional_set.add(item)
    set_add_time = time.time() - start_time
    
    # Test existing items
    print("\nTesting existing items:")
    existing_in_bf = sum(1 for item in test_data if bf.might_contain(item))
    existing_in_set = sum(1 for item in test_data if traditional_set.contains(item))
    
    print(f"Bloom Filter found: {existing_in_bf}/{len(test_data)}")
    print(f"Traditional Set found: {existing_in_set}/{len(test_data)}")
    
    # Test non-existing items (false positives)
    print("\nTesting non-existing items:")
    false_positives = sum(1 for item in non_existent_data if bf.might_contain(item))
    false_positives_set = sum(1 for item in non_existent_data if traditional_set.contains(item))
    
    actual_fp_rate = false_positives / len(non_existent_data)
    
    print(f"Bloom Filter false positives: {false_positives}/{len(non_existent_data)} ({actual_fp_rate:.2%})")
    print(f"Traditional Set false positives: {false_positives_set}/{len(non_existent_data)}")
    
    # Statistics
    stats = bf.get_statistics()
    print(f"\nSTATISTICS:")
    print(f"Bloom Filter size: {stats['size']} bits ({stats['size'] // 8} bytes)")
    print(f"Traditional Set size: {traditional_set.size_in_bytes()} bytes")
    print(f"Size ratio: {traditional_set.size_in_bytes() / (stats['size'] // 8):.1f}x")
    print(f"Bit array fill ratio: {stats['fill_ratio']:.2%}")
    print(f"Expected false positive rate: {stats['expected_fp_rate']:.2%}")
    print(f"Actual false positive rate: {stats['current_fp_rate']:.2%}")
    
    print(f"\nEXECUTION TIME:")
    print(f"Bloom Filter addition: {bf_add_time:.4f}s")
    print(f"Traditional Set addition: {set_add_time:.4f}s")
    
    return bf, stats, actual_fp_rate

def analyze_false_positive_rates():
    """
    Analyze how false positive rate depends on number of items
    """
    print("\n" + "=" * 60)
    print("FALSE POSITIVE RATES ANALYSIS")
    print("=" * 60)
    
    sizes = [100, 500, 1000, 2000, 5000]
    fp_rates = []
    theoretical_rates = []
    
    for size in sizes:
        bf = BloomFilter(expected_items=1000, false_positive_rate=0.01)
        
        # Add different number of items
        for i in range(size):
            bf.add(f"item_{i}")
        
        # Test false positive rate
        test_items = [f"fake_{i}" for i in range(1000)]
        false_positives = sum(1 for item in test_items if bf.might_contain(item))
        actual_rate = false_positives / len(test_items)
        
        fp_rates.append(actual_rate)
        theoretical_rates.append(bf.current_false_positive_rate())
        
        print(f"Size {size}: Actual FP rate: {actual_rate:.3f}, "
              f"Theoretical: {bf.current_false_positive_rate():.3f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, fp_rates, 'bo-', label='Actual FP rate', linewidth=2)
    plt.plot(sizes, theoretical_rates, 'ro--', label='Theoretical FP rate', linewidth=2)
    plt.axhline(y=0.01, color='g', linestyle=':', label='Target FP rate (1%)')
    
    plt.xlabel('Number of added items')
    plt.ylabel('False Positive Rate')
    plt.title('False Positive Rate dependency on number of items')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def spell_checker_example():
    """
    Usage example: Spell Checker
    """
    print("\n" + "=" * 60)
    print("EXAMPLE: SPELL CHECKER WITH BLOOM FILTER")
    print("=" * 60)
    
    # Dictionary with English words (simplified)
    english_words = [
        "algorithm", "computer", "programming", "software", "hardware", "internet",
        "database", "network", "security", "encryption", "data", "structure",
        "analysis", "machine", "learning", "artificial", "intelligence", "science",
        "technology", "development", "engineering", "mathematics", "statistics",
        "probability", "filter", "bloom", "hash", "function", "memory", "storage",
        "optimization", "performance", "efficiency", "scalability", "reliability",
        "availability", "consistency", "distributed", "system", "architecture",
        "design", "pattern", "framework", "library", "application", "interface"
    ]
    
    print(f"Creating dictionary with {len(english_words)} words...")
    
    # Bloom Filter dictionary
    bf_dict = BloomFilter(expected_items=len(english_words), false_positive_rate=0.01)
    for word in english_words:
        bf_dict.add(word.lower())
    
    # Traditional dictionary
    traditional_dict = set(word.lower() for word in english_words)
    
    # Test words
    test_words = ["algorithm", "puppy", "programming", "xyz123", "mathematics", "qwerty"]
    
    print(f"\nTesting words: {test_words}")
    print("\nResults:")
    print("-" * 50)
    
    for word in test_words:
        bf_result = bf_dict.might_contain(word.lower())
        traditional_result = word.lower() in traditional_dict
        
        status = "✓" if traditional_result else "✗"
        bf_status = "MIGHT BE" if bf_result else "NOT IN"
        
        print(f"{word:12} | Reality: {status} | Bloom Filter: {bf_status}")
        
        if bf_result and not traditional_result:
            print(f"             | ⚠️  FALSE POSITIVE!")
    
    # Efficiency analysis
    stats = bf_dict.get_statistics()
    traditional_size = sum(len(word.encode('utf-8')) for word in english_words)
    
    print(f"\nEFFICIENCY ANALYSIS:")
    print(f"Bloom Filter size: {stats['size'] // 8} bytes")
    print(f"Traditional dictionary: {traditional_size} bytes")
    print(f"Space saved: {((traditional_size - stats['size'] // 8) / traditional_size) * 100:.1f}%")

def performance_comparison():
    """
    Compare performance of different structures
    """
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    sizes = [1000, 5000, 10000, 50000]
    bf_times = []
    set_times = []
    
    # Increase number of tests for more accurate measurement
    num_tests = 10000
    
    for size in sizes:
        print(f"\nTesting with {size} items...")
        
        # Prepare data
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
        
        # Avoid division by zero
        if bf_time > 0:
            ratio = set_time / bf_time
            print(f"Ratio: {ratio:.2f}x")
        else:
            print("Ratio: N/A (too fast to measure)")
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(sizes, bf_times, 'bo-', label='Bloom Filter', linewidth=2)
    plt.plot(sizes, set_times, 'ro-', label='Traditional Set', linewidth=2)
    plt.xlabel('Number of items in structure')
    plt.ylabel(f'Time for {num_tests} lookups (seconds)')
    plt.title('Membership test time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Avoid division by zero when calculating speedup
    speedup = []
    for i in range(len(sizes)):
        if bf_times[i] > 0:
            speedup.append(set_times[i] / bf_times[i])
        else:
            speedup.append(1.0)  # If time is 0, assume no speedup
    
    plt.plot(sizes, speedup, 'go-', linewidth=2)
    plt.xlabel('Number of items in structure')
    plt.ylabel('Speedup (times)')
    plt.title('Bloom Filter speedup vs Set')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Theoretical analysis
def theoretical_analysis():
    """
    Theoretical analysis of Bloom Filter
    """
    print("\n" + "=" * 60)
    print("THEORETICAL ANALYSIS")
    print("=" * 60)
    
    print("""
    1. WHAT IS A DATA STRUCTURE?
    A data structure is a way of organizing and storing data
    that allows efficient access and modification.
    
    2. WHAT IS A PROBABILISTIC DATA STRUCTURE?
    A probabilistic data structure uses randomization in its operations
    or may give inaccurate results with known probability.
    
    3. BLOOM FILTER - CONSTRUCTION:
    - Bit array with m positions (initially all 0)
    - k independent hash functions
    - To add element: hash with all k functions and set 1 at corresponding positions
    - To check: hash with all k functions and verify all positions are 1
    
    4. OPERATIONS AND COMPLEXITY:
    - Addition: O(k) - constant time
    - Membership test: O(k) - constant time
    - Deletion: NOT POSSIBLE (destructive for other elements)
    
    5. PROBABILITIES:
    - False Positive: Possible (structure says "might be", but actually isn't)
    - False Negative: IMPOSSIBLE (if structure says "not", then definitely not)
    - False Positive probability: (1 - e^(-kn/m))^k
    
    6. ADVANTAGES:
    - Very small memory footprint (O(m) bits)
    - Fast operations O(k)
    - Doesn't store actual data (privacy)
    
    7. DISADVANTAGES:
    - False positives
    - Inability to delete
    - No access to actual elements
    
    8. USAGE:
    - Spell checkers
    - Web crawlers (checking visited URLs)
    - Database systems (avoiding expensive disk operations)
    - Cache systems
    - Network routers
    """)

def main():
    """
    Main function to execute all analyses
    """
    print("ANALYSIS OF PROBABILISTIC DATA STRUCTURES")
    print("Focus: Bloom Filter")
    print("=" * 60)
    
    # Theoretical analysis
    theoretical_analysis()
    
    # Demonstration
    bf, stats, fp_rate = demonstrate_bloom_filter()
    
    # False positive rates analysis
    analyze_false_positive_rates()
    
    # Usage example
    spell_checker_example()
    
    # Performance comparison
    performance_comparison()
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("""
    Bloom Filter is a powerful probabilistic data structure that:
    
    ✅ ADVANTAGES:
    - Extremely memory efficient (10-20x less space)
    - Fast operations (constant time)
    - Suitable for large-scale data systems
    - Guarantees no false negatives
    
    ⚠️ DISADVANTAGES:
    - Possible false positives (controllable)
    - Inability to delete elements
    - Doesn't store actual data
    
    🎯 USAGE:
    Ideal for cases where:
    - Memory is limited
    - False positives are acceptable
    - Very fast membership testing is needed
    - Data volume is large
    
    Examples: spell checkers, web crawlers, cache systems, network routers.
    """)

if __name__ == "__main__":
    main()
