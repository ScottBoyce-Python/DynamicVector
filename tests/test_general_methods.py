import pytest
import numpy as np
from DynamicVector import DynamicVector


@pytest.fixture
def empty_vector():
    return DynamicVector(capacity=2)


@pytest.fixture
def small_vector():
    vec = DynamicVector(capacity=4)
    for i in range(5):
        vec.append(i)
    return vec


@pytest.fixture
def large_vector():
    vec = DynamicVector(capacity=64)
    for i in range(100):
        vec.append(i)
    return vec


def test_initial_capacity():
    vec = DynamicVector(capacity=16)
    assert vec.capacity == 16
    assert vec.size == 0


def test_dynamic_growth():
    vec = DynamicVector(capacity=2)
    vec.extend([1, 2, 3, 4])
    assert vec.capacity > 2  # Verify dynamic growth
    assert vec.size == 4


def test_force_capacity(small_vector):
    initial_capacity = small_vector.capacity
    small_vector.force_capacity(100)
    assert small_vector.capacity >= 100
    assert small_vector.capacity != initial_capacity


def test_contains(small_vector):
    assert 3 in small_vector
    assert 99 not in small_vector


def test_resize_larger(small_vector):
    small_vector.resize(10)
    assert small_vector.size == 10


def test_resize_smaller(small_vector):
    small_vector.resize(2)
    assert small_vector.size == 2


def test_copy(small_vector):
    copy_vec = small_vector.copy()
    assert copy_vec.size == small_vector.size
    assert np.array_equal(copy_vec.view, small_vector.view)


def test_invalid_index(small_vector):
    with pytest.raises(IndexError):
        small_vector[10]


def test_abs(large_vector):
    large_vector.abs()
    assert np.array_equal(large_vector.view, np.abs(large_vector.view))


def test_is_equal(small_vector):
    other = DynamicVector.from_values([0, 1, 2, 3, 4])
    assert small_vector.is_equal(other)


def test_add_operator():
    vec1 = DynamicVector.from_values([1, 2, 3])
    vec2 = DynamicVector.from_values([4, 5, 6])
    result = vec1 + vec2
    assert np.array_equal(result.view, np.array([5, 7, 9]))


def test_sub_operator():
    vec1 = DynamicVector.from_values([10, 20, 30])
    vec2 = DynamicVector.from_values([1, 2, 3])
    result = vec1 - vec2
    assert np.array_equal(result.view, np.array([9, 18, 27]))


def test_invalid_slice(small_vector):
    with pytest.raises(IndexError):
        small_vector[10:20]


def test_dtype():
    vec = DynamicVector(dtype=float)
    vec.append(1.5)
    assert vec.dtype == np.float64


def test_view(small_vector):
    view = small_vector.view
    assert isinstance(view, np.ndarray)
    assert len(view) == small_vector.size


def test_clear1():
    vector = DynamicVector.from_values([1, 2, 3])
    vector.clear()
    assert vector.is_equal([])


def test_clear2(small_vector):
    small_vector.clear()
    assert small_vector.size == 0
    assert len(small_vector.view) == 0


def test_len():
    vector = DynamicVector.from_values([1, 2, 3])
    assert len(vector) == 3


def test_index1():
    vector = DynamicVector.from_values([1, 2, 3, 4])
    assert vector.index(3) == 2


def test_index2(small_vector):
    assert small_vector.index(3) == 3
    with pytest.raises(ValueError):
        small_vector.index(99)


def test_reverse1():
    vector = DynamicVector.from_values([1, 2, 3, 4])
    vector.reverse()
    assert vector.is_equal([4, 3, 2, 1])


def test_reverse2(small_vector):
    small_vector.reverse()
    assert small_vector[0] == 4
    assert small_vector[-1] == 0


def test_sort1():
    vector = DynamicVector.from_values([3, 1, 4, 2])
    vector.sort()
    assert vector.is_equal([1, 2, 3, 4])


def test_sort2(small_vector):
    small_vector.extend([99, -10, 50])
    small_vector.sort()
    assert small_vector.view[0] == -10
    assert small_vector.view[-1] == 99
