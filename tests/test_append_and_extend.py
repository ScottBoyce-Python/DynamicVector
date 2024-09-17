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


def test_append():
    vector = DynamicVector.from_values([1, 2, 3])
    vector.append(4)
    assert vector.is_equal([1, 2, 3, 4])


def test_append1(small_vector):
    small_vector.append(5)
    assert small_vector.size == 6
    assert small_vector[5] == 5


def test_append32(small_vector):
    for i in range(5, 32):
        small_vector.append(i)
    assert small_vector.size == 32
    assert small_vector[5] == 5
    assert small_vector[30] == 30


def test_extend():
    vector = DynamicVector.from_values([1, 2, 3])
    vector.extend([4, 5])
    assert vector.is_equal([1, 2, 3, 4, 5])


def test_extend1(small_vector):
    small_vector.extend([5, 6, 7])
    assert small_vector.size == 8
    assert small_vector[5] == 5
    assert small_vector[7] == 7


def test_extend2(small_vector):
    small_vector.extend([5, 6, 7])
    small_vector.extend(range(8, 32))
    assert small_vector.size == 32
    assert small_vector[5] == 5
    assert small_vector[7] == 7
    assert small_vector[30] == 30


def test_extend3(small_vector):
    small_vector.extend([5, 6, 7])
    small_vector.extend(range(8, 32))
    small_vector.extend([32, 33, 34, 35, 36, 37, 38, 39, 40])
    assert small_vector.size == 41
    assert small_vector[5] == 5
    assert small_vector[7] == 7
    assert small_vector[30] == 30
    assert small_vector[40] == 40
