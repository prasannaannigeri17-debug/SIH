"""
Unit tests for Benchmarking Mathematics.
"""
from benchmarking.metrics import PerformanceMetrics


def test_reduction_formulas():
    # 10,000 baseline cells reduced to 2,500 adaptive cells -> 75% reduction
    cell_red = PerformanceMetrics.calculate_cell_reduction(10000, 2500)
    assert cell_red == 75.0

    # 1000 KB baseline reduced to 300 KB adaptive -> 70% reduction
    mem_red = PerformanceMetrics.calculate_memory_reduction(1000, 300)
    assert mem_red == 70.0
