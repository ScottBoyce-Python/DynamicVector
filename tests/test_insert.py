import pytest
from DynamicVector import DynamicVector


@pytest.fixture
def small_vector():
    vec = DynamicVector(capacity=4)
    for i in range(5):
        vec.append(i)
    return vec


def test_insert1():
    vector = DynamicVector.from_vector([1, 2, 4])
    vector.insert(2, 3)
    assert vector.is_equal([1, 2, 3, 4])


def test_insert2(small_vector):
    small_vector.insert(2, 99)
    assert small_vector.size == 6
    assert small_vector[2] == 99


def test_insert_at_start(small_vector):
    small_vector.insert(0, 88)
    assert small_vector.size == 6
    assert small_vector[0] == 88


def test_insert_at_end(small_vector):
    small_vector.insert(len(small_vector), 88)
    assert small_vector.size == 6
    assert small_vector[-1] == 88


def test_insert_values_at_start(small_vector):
    small_vector.insert_values(0, [99, 100])
    assert small_vector.size == 7
    assert small_vector[0] == 99
    assert small_vector[1] == 100


def test_insert_values_at_middle(small_vector):
    small_vector.insert_values(2, [55, 66])
    assert small_vector.size == 7
    assert small_vector[2] == 55
    assert small_vector[3] == 66


def test_insert_values_at_end(small_vector):
    small_vector.insert_values(small_vector.size, [88, 99])
    assert small_vector.size == 7
    assert small_vector[-2] == 88
    assert small_vector[-1] == 99


def test_insert_single_value(small_vector):
    small_vector.insert_values(2, [77])
    assert small_vector.size == 6
    assert small_vector[2] == 77


def test_insert_values_dynamic_growth(small_vector):
    small_vector.insert_values(1, [100] * 100)
    assert small_vector.size == 105
    assert small_vector[1] == 100
    assert small_vector[100] == 100
    assert small_vector[101] == 1  # Original value at index 1 shifted to 101


def test_insert_values_out_of_bounds(small_vector):
    with pytest.raises(IndexError):
        small_vector.insert_values(10, [1, 2, 3])


def test_insert_empty_values(small_vector):
    small_vector.insert_values(2, [])
    assert small_vector.size == 5  # No change in size when inserting empty list
