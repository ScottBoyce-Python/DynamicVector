import pytest
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


def test_remove():
    vector = DynamicVector.from_values([1, 2, 3, 4])
    vector.remove(3)
    assert vector.is_equal([1, 2, 4])


def test_remove_value(small_vector):
    assert small_vector.remove(3)
    assert small_vector.size == 4
    assert 3 not in small_vector.view


def test_remove_all(small_vector):
    small_vector.extend([2, 2, 2])
    assert small_vector.remove(2, remove_all=True)
    assert small_vector.size == 4
    assert 2 not in small_vector


def test_pop1():
    vector = DynamicVector.from_values([1, 2, 3])
    popped_value = vector.pop()
    assert popped_value == 3
    assert vector.is_equal([1, 2])


def test_pop2(small_vector):
    val = small_vector.pop()
    assert val == 4
    assert small_vector.size == 4


def test_pop_empty(empty_vector):
    with pytest.raises(IndexError):
        empty_vector.pop()
