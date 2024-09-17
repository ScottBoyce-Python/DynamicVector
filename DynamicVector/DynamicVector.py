import numpy as np

# Constants that determine when to switch from multiplicative to additive growth
_GROW_USE_ADD = 2**13  # Threshold capacity (8192), where growth switches to additive mode
_GROW_ADD = 2**11  # Capacity to add when additive mode is active (2048)


class DynamicVector:
    """
    A dynamic vector implementation using NumPy arrays, with dynamic resizing capabilities.
    The vector grows in capacity using powers of two until it exceeds a certain threshold,
    after which it grows by a fixed amount.

    Attributes:
        _size (int): The current number of elements in the vector.
        _cap (int): The current capacity of the underlying array.
        _data (np.ndarray): The underlying NumPy array that stores the elements.
        _dtype (np.dtype): The data type of the array elements.
    """

    _size: int
    _cap: int
    _data: np.ndarray
    _dtype: np.dtype

    def __init__(self, dtype=np.int32, capacity=8, *, grow_use_add=None, grow_add=None):
        """
        Initialize the DynamicVector.

        Parameters:
            dtype (np.dtype): The data type of the elements (int32 by default).
            capacity (int): The initial capacity of the vector. Defaults to 8.
            grow_use_add (int, optional): Custom threshold to switch from multiplicative to additive growth.
            grow_add (int, optional): Custom value for additive growth.
        """
        if grow_use_add is None:
            self._grow_use_add = _GROW_USE_ADD
        else:
            self._grow_use_add = self._next_power_of_2(grow_use_add)

        if grow_add is None:
            self._grow_add = _GROW_ADD
        else:
            self._grow_add = self._next_power_of_2(grow_add)

        if dtype is int:
            dtype = np.int32
        if dtype is float:
            dtype = np.float64

        self._size = 0

        self._cap = 2
        self._setup_capacity(capacity)  # increase self._cap to meet capacity
        self._data = np.zeros(self._cap, dtype=dtype)
        self._dtype = self._data.dtype
        self._zero = self._data[0]

    @classmethod
    def from_vector(cls, vector, *, grow_use_add=None, grow_add=None):
        """
        Create a DynamicVector from an existing vector.

        Parameters:
            vector (sequence): The source array to initialize the vector.
            grow_use_add (int, optional): Custom threshold to switch from multiplicative to additive growth.
            grow_add (int, optional): Custom value for additive growth.

        Returns:
            DynamicVector: A new dynamic vector initialized with the values from the input vector.
        """
        try:
            capacity = len(vector)
        except TypeError:
            return cls.from_iter(vector, grow_use_add, grow_add)

        try:
            dtype = vector.dtype
        except AttributeError:
            try:
                dtype = type(vector[0])
            except IndexError:
                raise ValueError("Either pass variable with dtype attribute, or len(vector) must be greater than zero.")

        if isinstance(vector, DynamicVector):
            if grow_use_add is None:
                grow_use_add = vector.grow_use_add
            if grow_add is None:
                grow_add = vector.grow_add

        dyn = cls(dtype, capacity, grow_use_add=grow_use_add, grow_add=grow_add)
        dyn.extend(vector)
        return dyn

    @classmethod
    def from_iter(cls, iterator, *, grow_use_add=None, grow_add=None):
        """
        Create a DynamicVector from an iterator.

        Parameters:
            iterator (iterator): The source iterator to initialize the vector.
            grow_use_add (int, optional): Custom threshold to switch from multiplicative to additive growth.
            grow_add (int, optional): Custom value for additive growth.

        Returns:
            DynamicVector: A new dynamic vector initialized with the values from the input iterator.
        """
        try:
            value = next(iterator)
        except TypeError:
            iterator = iter(iterator)
            value = next(iterator)

        try:
            dtype = value.dtype
        except AttributeError:
            try:
                dtype = type(value)
            except IndexError:
                raise ValueError("Iterator failed to identify dtype")

        if isinstance(iterator, DynamicVector):
            if grow_use_add is None:
                grow_use_add = iterator.grow_use_add
            if grow_add is None:
                grow_add = iterator.grow_add

        dyn = cls(dtype, grow_use_add=grow_use_add, grow_add=grow_add)
        dyn.append(value)
        for value in iterator:
            dyn.append(value)
        return dyn

    @property
    def size(self) -> int:
        """Returns the current size of the vector."""
        return self._size

    @property
    def capacity(self) -> int:
        """Returns the current capacity of the vector."""
        return self._cap

    @property
    def view(self) -> np.ndarray:
        """Returns a numpy array view of the vector at its current size.
        This view can use all the numpy built in methods.
        Any changes to the values in the DynamicVector are reflected in the view,
        and vice-versa.

        The view does NOT change if the DynamicVector size changes.
        If the size does change, then you must remake the view.

        It is recommended to not set view to another variable,
        but instead use `self.view` it as needed."""
        return self._data[: self._size]

    @property
    def dtype(self) -> np.dtype:
        """Returns the data type of the vector."""
        return self._dtype

    @property
    def grow_use_add(self) -> int:
        """Returns the threshold capacity where growth switches to additive."""
        return self._grow_use_add

    @property
    def grow_add(self) -> int:
        """Returns the capacity increment used in additive growth mode."""
        return self._grow_add

    def append(self, value):
        """
        Append a value to the end of the vector, increasing the vector's size by one.

        Parameters:
            value: The value to be appended.
        """
        if self._size >= self._cap:
            self._grow_data(self._size + 1)

        self._data[self._size] = value
        self._size += 1

    def extend(self, values):
        """
        Extend the vector by appending multiple values.
        The size of the vector increases to reflect the addition of the values.

        Parameters:
            values (iterable): The values to append to the vector.
        """
        try:
            new_size = self._size + len(values)
        except TypeError:  # iterator so need to manually add values
            for value in values:
                self.append(value)
            return

        if new_size >= self._cap:
            self._grow_data(new_size)

        self._data[self._size : new_size] = values
        self._size = new_size

    def insert(self, index, value):
        """Insert an item at a given position.
        The first argument is the index of the element before which to insert, so a.insert(0, x)
        inserts at the front of the list, and a.insert(len(a), x) is equivalent to a.append(x)."""
        if isinstance(index, int):
            if index == self._size:
                self.append(value)
                return

            index = [self._format_int_index(index)]

        elif isinstance(index, slice):
            index = self._slice_to_range(index)
        else:
            index = sorted(index, reverse=True)  # assume its listlike input

        if self._size + len(index) >= self._cap:
            self._grow_data(self._size + len(index))

        for p in index:
            self._size += 1
            self._data[index + 1 : self._size] = self._data[index : self._size - 1]
            self._data[index] = value

    def remove(self, value, remove_all=False, from_right=False) -> bool:
        # remove value from array, return true if one or more values found
        if self._size < 1:
            return False

        index = self.where(value)
        if len(index) < 1:
            return False  # no values found

        if remove_all:
            index = index[::-1]
        elif from_right:
            index = index[-1:]
        else:
            index = index[:1]

        for p in index:
            self.drop(p)

        return True

    def pop(self, index=-1, return_None=False):
        if isinstance(index, int):
            if self._size < 1:
                if return_None:
                    return None
                raise IndexError("pop from empty vector")

            index = self._format_int_index(index)

            value = self._data[index]

            self.drop(index)

            return value

        if isinstance(index, slice):
            popped = [self._data[p] for p in self._slice_to_range(index)]
            index = self._slice_to_range(index)
        else:
            popped = [self._data[p] for p in index]
            index = sorted(index, reverse=True)

        for p in index:
            self.drop(p)

        return popped

    def drop(self, index=-1):
        if self._size < 1:
            return

        if isinstance(index, int):
            if index < 0:
                index = self._size + index
            if index < 0 or index >= self._size:
                return
            if index == self._size - 1:
                self._size -= 1
                return
            index = [index]

        elif isinstance(index, slice):
            index = self._slice_to_range(index)
        else:
            # assume its list-like input
            index = sorted([self._size + p if p < 0 else p for p in index], reverse=True)

        for p in index:
            if 0 <= p < self._size:
                self._size -= 1
                if p < self._size:
                    self._data[p : self._size] = self._data[p + 1 : self._size + 1]

    def count(self, value) -> int:
        return len(self.where(value))

    def sort(self, reverse=False, kind=None):
        """Sort array in place in ascending order.
        kind can be set to quicksort, heapsort, or stable
        Sorting algorithm. The default is quicksort.
        reverse returns sorted array in descending order
        """
        if self._size < 2:
            return
        self._data[: self._size].sort(kind=kind)

        if reverse:
            self._reverse_in_place(self._data[: self._size])

    def reverse(self):
        self._reverse_in_place(self._data[: self._size])

    def contains(self, value) -> bool:
        return value in self._data[: self._size]

    def __contains__(self, value):
        return value in self._data[: self._size]

    def copy(self, min_capacity=8):
        capacity = self._size if min_capacity < self._size else min_capacity
        dyn = DynamicVector(self._dtype, capacity, self._grow_use_add, self._grow_add)
        dyn.extend(self._data[: self._size])
        return dyn

    def clear(self):
        self._size = 0

    def abs(self, where=True):
        vec = self._data[: self._size]
        np.absolute(vec, out=vec, where=where)

    def where(self, value):
        return np.where(self._data[: self._size] == value)[0]

    def resize(self, size: int):
        if size < self._size:
            self._data[size : self._size] = self._zero
        elif size > self._cap:
            self._grow_data(size)
            self._size = size
        else:
            self._size = size

    def increase_size(self, increase_by: int):
        self.resize(self._size + increase_by)

    def set_capacity(self, min_capacity):
        # set capacity to a power of 2 that exceeds min_capacity
        if min_capacity > self._cap:
            self._grow_data(min_capacity)

    def force_capacity(self, min_capacity):
        # set capacity be smallest power of two that exceeds min_capacity
        if min_capacity == self._cap:
            return

        old_cap = self._cap
        self._cap = self._next_power_of_2(min_capacity) // 2  # find closes power of 2 that is less than min_capacity

        self._setup_capacity(min_capacity)  # determine proper capacity that exceeds min_capacity

        if old_cap == self._cap:  # ended up with same capacity, so nothing to do
            return

        tmp = np.zeros(self._cap, dtype=self._dtype)

        if old_cap < self._cap:  # resulted in larger new array
            tmp[:old_cap] = self._data
        else:
            tmp[:] = self._data[: self._cap]  # resulted in smaller new array

        self._data = tmp

    def __getitem__(self, index):
        if isinstance(index, int):
            if index < 0:
                index = self._size + index
            if index < 0 or index >= self._size:
                raise IndexError(f"Index out of bounds. {index} >= {self._size}")

        elif isinstance(index, slice):
            start = 0 if index.start is None else index.start
            stop = self._size if index.stop is None else index.stop
            step = 1 if index.step is None else index.step

            if start < 0:
                start = self._size + start
            if stop < 0:
                stop = self._size + stop

            if start < 0 or stop > self._size or stop < 0:
                raise IndexError("Slice out of bounds")
            index = slice(start, stop, step)

        return self._data[: self._size][index]

    def __setitem__(self, index, value):
        if isinstance(index, int):
            if index < 0:
                index = self._size + index
            if index < 0 or index >= self._size:
                raise IndexError(f"Index out of bounds. {index} >= {self._size}")

        elif isinstance(index, slice):
            start = 0 if index.start is None else index.start
            stop = self._size if index.stop is None else index.stop
            step = 1 if index.step is None else index.step

            if start < 0:
                start = self._size + start
            if stop < 0:
                stop = self._size + stop

            if start < 0 or stop > self._size or stop < 0:
                raise IndexError("Slice out of bounds")
            index = slice(start, stop, step)

        self._data[: self._size][index] = value

    def __repr__(self):
        return repr(self.view)

    def __str__(self):
        return str(self.view)

    def __iter__(self):
        return iter(self.view)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, unused=None):
        return self.copy()

    def __pos__(self):  # +val
        return self.copy()

    def __neg__(self):  # -val
        res = self.copy()
        res.view *= -1.0
        return res

    def __abs__(self):  # abs(val)
        res = self.copy()
        res.abs()
        return res

    def __iadd__(self, other):  # To get called on addition with assignment e.g. a +=b.
        if isinstance(other, DynamicVector):
            other = other.view
        self.view += other

    def __isub__(self, other):  # To get called on subtraction with assignment e.g. a -=b.
        if isinstance(other, DynamicVector):
            other = other.view
        self.view -= other

    def __imul__(self, other):  # To get called on multiplication with assignment e.g. a *=b.
        if isinstance(other, DynamicVector):
            other = other.view
        self.view *= other

    def __itruediv__(self, other):  # To get called on true division with assignment e.g. a /=b.
        if isinstance(other, DynamicVector):
            other = other.view
        self.view /= other

    def __irtruediv__(self, other):  # To get called on true division with assignment e.g. a /=b.
        if isinstance(other, DynamicVector):
            other = other.view
        self.view /= other

    def __ifloordiv__(self, other):  # To get called on integer division with assignment e.g. a //=b.self.view /= other
        if isinstance(other, DynamicVector):
            other = other.view
        self.view //= other

    def __irfloordiv__(self, other):  # To get called on integer division with assignment e.g. a //=b.
        if isinstance(other, DynamicVector):
            other = other.view
        self.view //= other

    def __imod__(self, other):  # To get called on modulo with assignment e.g. a%=b.
        if isinstance(other, DynamicVector):
            other = other.view
        self.view %= other

    def __irmod__(self, other):  # To get called on modulo with assignment e.g. a%=b.
        if isinstance(other, DynamicVector):
            other = other.view
        self.view %= other

    def __ipow__(self, other):  # To get called on exponents with assignment e.g. a **=b.
        if isinstance(other, DynamicVector):
            other = other.view
        self.view **= other

    def __irpow__(self, other):  # To get called on exponents with assignment e.g. a **=b.
        if isinstance(other, DynamicVector):
            other = other.view
        self.view **= other

    def __int__(self):  # To get called by built-int int() method to convert a type to an int.
        res = DynamicVector(int, self._cap)
        res.extend(self.view)
        return res

    def __float__(self):  # To get called by built-int float() method to convert a type to float.
        res = DynamicVector(float, self._cap)
        res.extend(self.view)
        return res

    def __add__(self, other):  # To get called on add operation using + operator
        res = self.copy()
        res += other
        return res

    def __radd__(self, other):  # To get called on add operation using + operator
        res = self.copy()
        res += other
        return res

    def __sub__(self, other):  # To get called on subtraction operation using - operator.
        res = self.copy()
        res -= other
        return res

    def __rsub__(self, other):  # To get called on subtraction operation using - operator.
        res = self.copy()
        res -= other
        return res

    def __mul__(self, other):  # To get called on multiplication operation using * operator.
        res = self.copy()
        res *= other
        return res

    def __rmul__(self, other):  # To get called on multiplication operation using * operator.
        res = self.copy()
        res *= other
        return res

    def __floordiv__(self, other):  # To get called on floor division operation using // operator.
        res = self.copy()
        res //= other
        return res

    def __rfloordiv__(self, other):  # To get called on floor division operation using // operator.
        res = self.copy()
        res //= other
        return res

    def __truediv__(self, other):  # To get called on division operation using / operator.
        res = self.copy()
        res /= other
        return res

    def __rtruediv__(self, other):  # To get called on division operation using / operator.
        res = self.copy()
        res /= other
        return res

    def __mod__(self, other):  # To get called on modulo operation using % operator.
        res = self.copy()
        res %= other
        return res

    def __rmod__(self, other):  # To get called on modulo operation using % operator.
        res = self.copy()
        res %= other
        return res

    def __pow__(self, other):  # To get called on calculating the power using ** operator.
        res = self.copy()
        res **= other
        return res

    def __rpow__(self, other):  # To get called on calculating the power using ** operator.
        res = self.copy()
        res **= other
        return res

    def __lt__(self, other):  # To get called on comparison using < operator.
        if isinstance(other, DynamicVector):
            other = other.view
        return self.view < other

    def __le__(self, other):  # To get called on comparison using <= operator.
        if isinstance(other, DynamicVector):
            other = other.view
        return self.view <= other

    def __gt__(self, other):  # To get called on comparison using > operator.
        if isinstance(other, DynamicVector):
            other = other.view
        return self.view > other

    def __ge__(self, other):  # To get called on comparison using >= operator.
        if isinstance(other, DynamicVector):
            other = other.view
        return self.view >= other

    def __eq__(self, other):  # To get called on comparison using == operator.
        if isinstance(other, DynamicVector):
            other = other.view
        return self.view == other

    def __ne__(self, other):  # To get called on comparison using != operator.
        if isinstance(other, DynamicVector):
            other = other.view
        return self.view != other

    def _format_int_index(self, index: int) -> int:
        if index < 0:
            index = self._size + index
        if index < 0 or index >= self._size:
            raise IndexError(f"Index out of bounds. {index} >= {self._size}")
        return index

    def _slice_to_range(self, sl: slice, same_order=False):
        # given a slice, convert to descending order range
        start = 0 if sl.start is None else sl.start
        stop = self._size if sl.stop is None else sl.stop
        step = 1 if sl.step is None else sl.step

        if start < 0:
            start = self._size + start
        if stop < 0:
            stop = self._size + stop

        if start < 0 or stop > self._size or stop < 0:
            raise IndexError("Slice out of bounds")

        if same_order:
            return range(start, stop, step)

        if step < 0:
            return range(start, stop, step)
        return range(stop - 1, start - 1, -step)

    def _grow_data(self, min_capacity):
        self._setup_capacity(min_capacity)
        old_cap = len(self._data)
        if old_cap < self._cap:  # should always be true
            tmp = np.zeros(self._cap, dtype=self._dtype)
            tmp[:old_cap] = self._data
            self._data = tmp
        else:
            self._cap = old_cap

    def _setup_capacity(self, min_capacity):
        while self._cap < min_capacity:
            if self._cap < self._grow_use_add:
                self._cap <<= 1
            else:
                self._cap += self._grow_add

    @staticmethod
    def _next_power_of_2(x: int):
        """
        Calculate the next power of 2 greater than or equal to x.

        Parameters:
            x (int): The input number.

        Returns:
            int: The next power of 2.
        """
        if x < 2:
            return 2
        x -= 1
        x |= x >> 1
        x |= x >> 2
        x |= x >> 4
        x |= x >> 8
        x |= x >> 16
        x |= x >> 32
        return x + 1

    @staticmethod
    def _reverse_in_place(array):
        """
        Reverse the contents of the given array in place.

        Parameters:
            array (np.ndarray): The array to reverse.
        """
        n = array.shape[0]
        for i in range(n // 2):
            array[i], array[n - i - 1] = array[n - i - 1], array[i]


if __name__ == "__main__":
    x = DynamicVector()
    for i in range(32):
        x.append(i)
    print(x)

    x = DynamicVector.from_vector([1, 2, 3, 4, 5])
