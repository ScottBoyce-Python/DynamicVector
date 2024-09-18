import numpy as np

try:
    from ._metadata import (
        __version__,
        __author__,
        __email__,
        __license__,
        __status__,
        __maintainer__,
        __credits__,
        __url__,
        __description__,
        __copyright__,
    )
except ImportError:
    __version__ = "Failed to load from _metadata.py"
    __author__ = __version__
    __email__ = __version__
    __license__ = __version__
    __status__ = __version__
    __maintainer__ = __version__
    __credits__ = __version__
    __url__ = __version__
    __description__ = __version__
    __copyright__ = __version__

__all__ = [
    "DynamicVector",
]

# Constants that determine when to switch from multiplicative to additive growth
_GROW_USE_ADD = 2**13  # Threshold capacity (8192), where growth switches to additive mode
_GROW_ADD = 2**11  # Capacity to add when additive mode is active (2048)


class DynamicVector:
    """
    A dynamic vector implementation using NumPy arrays, with dynamic resizing capabilities.
    The dynamic vector includes fast appending and popping, like a list,
    while retaining numpy array index support and vector operations.

    The dynamic vector support the python list methods and numpy.ndarray 1D array methods.
    For access to all the numpy methods, a view of the vector as a numpy.ndarray can be returned.

    The storage of the vector automatically grows in capacity by doubling it
    until the capacity exceeds a certain threshold (`grow_use_add`),
    after which it grows by a fixed amount (`grow_add`).

    For example, given a dynamic vector that has an initial capacity of 8
    and contains seven values (size=7). If another value is appended (size=8),
    then the capacity is increased to 16 (from 2*8) to hold the extra value.

    Args:
        dtype (np.dtype, optional):   numpy.dtype (data type) of the vector elements. Defaults to np.int32.
        capacity (int, optional):     Initial minimum capacity of the underlying storage vector. Defaults to 8.
        grow_use_add (int, optional): Threshold to switch from multiplicative to additive growth. Defaults to 8192.
        grow_add (int, optional):     Additive increment in additive growth mode. Defaults to 2048.

    Attributes:
        size (int):         The size of the dynamic vector.
        view (np.ndarray):  A np.ndarray view of the dynamic vector at its current size.
        dtype (np.dtype):   The numpy.dtype (data type) of the vector.

        capacity (int):     The capacity of the underlying vector storage (note, `size <= capacity`).
        grow_use_add (int): Threshold that growth switches from multiplicative to additive growth.
        grow_add (int):     Additive increment in additive growth mode.

    Constructors:
        from_values(values): Create a DynamicVector from an iterable of values.

        from_iter(iterator): Create a DynamicVector from an iterator.

    Methods:

        append(value):
            Append a value to the end of the vector.

        extend(values):
            Append multiple values to the vector.

        insert(index, value):
            Insert a value at a specified index.

        insert_values(index, values):
            Insert multiple values at a specified index.

        pop(index=-1):
            Remove and return an element at a specified index (default is the last).

        remove(value, remove_all=False, from_right=False):
            Remove one or more occurrences of a value.

        drop(index=-1):
            Remove an element at a specified index.

        count(value):
            Count occurrences of a value.

        sort(reverse=False, kind=None):
            Sort the vector in ascending or descending order.

        reverse():
            Reverse the order of elements in the vector.

        contains(value):
            Check if a value exists in the vector.

        copy(min_capacity=8):
            Create a copy of the vector.

        clear():
            Remove all elements from the vector.

        abs(where=True):
            Compute the absolute value of all elements.

        where(value):
            Get the indices of elements equal to a value.

        index(value, from_right=False):
            Get the index of a value in the vector.

        resize(size):
            Resize the vector to a specified size.

        increase_size(increase_by):
            Increase the size of the vector by a specified amount.

        set_capacity(min_capacity):
            Ensure the vector's capacity is at least a given value.

        force_capacity(min_capacity):
            Set the capacity to the smallest power of two exceeding min_capacity.

        is_equal(other):
            Check if all elements are equal to those in another vector or value.

        is_less(other):
            Check if all elements are less than those in another vector or value.

        is_greater(other):
            Check if all elements are greater than those in another vector or value.

        is_less_or_equal(other):
            Check if all elements are less than or equal to those in another vector or value.

        is_greater_or_equal(other):
            Check if all elements are greater than or equal to those in another vector or value.

        is_not_equal(other):
            Check if all elements are not equal to those in another vector or value.


    Example usage:
        >>>
        >>> from DynamicVector import DynamicVector
        >>>
        >>> vec = DynamicVector(dtype=np.int32, capacity=4)
        >>>
        >>> vec.append(10)
        >>> vec.append(20)
        >>>
        >>> vec.extend([30, 40, 50])
        >>>
        >>> print(vec)
        DynamicVector([10, 20, 30, 40, 50])
        >>>
        >>> print(vec[2])     # vec[2] returns np.int32(20)
        30
        >>> print(vec[1:4])   # vec[1:4] returns a np.ndarray view of vector
        [20 30 40]
        >>>
        >>> vec.sort()        # inplace sort
        >>>
        >>> print(f"Size: {vec.size}, Capacity: {vec.capacity}")
        Size: 5, Capacity: 8
        >>>
        >>> vec.pop()         # Remove the last element
        >>>
        >>> vec.clear()       # Clear the vector (vec.size = 0)

    Notes:
    - The vector automatically resizes when needed, and it uses multiplicative growth until a specified threshold.
    - After reaching the threshold, the growth becomes additive.
    - A variable set to the view of the vector does not change size when the DynamicVector changes size.
    - Size is always less than or equal to capacity.
    """

    _size: int  # The current number of elements in the vector.
    _cap: int  # The current capacity of the underlying np.ndarray.
    _data: np.ndarray  # The underlying NumPy array that stores the elements.
    _dtype: np.dtype  # The data type of the array elements.

    def __init__(self, dtype=np.int32, capacity=8, *, grow_use_add=None, grow_add=None):
        """
        Initialize the DynamicVector.

        Parameters:
            dtype (np.dtype, optional):   numpy.dtype (data type) of the vector elements. Defaults to np.int32.
            capacity (int, optional):     Initial minimum capacity of the underlying storage vector. Defaults to 8.
            grow_use_add (int, optional): Threshold to switch from multiplicative to additive growth. Defaults to 8192.
            grow_add (int, optional):     Additive increment in additive growth mode. Defaults to 2048.
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
        # self._zero = self._data[0]

    @classmethod
    def from_values(cls, values, *, grow_use_add=None, grow_add=None):
        """
        Create a DynamicVector from an existing vector.

        Parameters:
            values (sequence): The source array to initialize the vector.
            grow_use_add (int, optional): Custom threshold to switch from multiplicative to additive growth.
            grow_add (int, optional): Custom value for additive growth.

        Returns:
            DynamicVector: A new dynamic vector initialized with the values from the input vector.
        """
        try:
            capacity = len(values)
        except TypeError:
            return cls.from_iter(values, grow_use_add, grow_add)

        try:
            dtype = values.dtype
        except AttributeError:
            try:
                dtype = type(values[0])
            except IndexError:
                raise ValueError("Either pass variable with dtype attribute, or len(values) must be greater than zero.")

        if isinstance(values, DynamicVector):
            if grow_use_add is None:
                grow_use_add = values.grow_use_add
            if grow_add is None:
                grow_add = values.grow_add

        dyn = cls(dtype, capacity, grow_use_add=grow_use_add, grow_add=grow_add)
        dyn.extend(values)
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

    # @view.setter
    # def view(self, value):
    #     self._data[: self._size] = value

    @property
    def dtype(self) -> np.dtype:
        """Returns the numpy.dtype (data type) of the vector."""
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

        Notes:
            1) Appending an item has a minimal performance penalty if size < capacity.
               If size == capacity, then the vector is reallocated to increase capacity.
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

        Notes:
            1) Extending an item has a minimal performance penalty if size + len(values) < capacity.
               If size + len(values) >= capacity, then the vector is reallocated to increase capacity.
        """
        try:
            new_size = self._size + len(values)
        except TypeError:  # iterator so need to manually add values
            for value in values:
                self.append(value)
            return

        if new_size > self._cap:
            self._grow_data(new_size)

        self._data[self._size : new_size] = values
        self._size = new_size

    def insert(self, index, value):
        """Insert an item at a given position.
        The first argument is the index of the element before which to insert, so a.insert(0, x)
        inserts at the front of the list, and a.insert(len(a), x) is equivalent to a.append(x).
        Index may be a single value or list-like array that represent all the index locations to place value"""
        if isinstance(index, (int, np.integer, np.unsignedinteger)):
            if index == self._size:
                self.append(value)
                return

            index = [self._format_int_index(index)]

        elif isinstance(index, slice):
            index = self._slice_to_range(index)
        else:
            index = sorted(index, reverse=True)  # assume its listlike input

        if self._size + len(index) > self._cap:
            self._grow_data(self._size + len(index))

        for p in index:
            self._size += 1
            self._data[p + 1 : self._size] = self._data[p : self._size - 1]
            self._data[p] = value

    def insert_values(self, index, values):
        """Insert a set of values at a given position.
        The first argument is the index of the element before which to insert, so a.insert(0, x)
        inserts at the front of the list, and a.insert(len(a), x) is equivalent to a.append(x).
        values may be any listlike array that supports len()."""

        if len(values) == 1:
            self.insert(index, values[0])
            return

        if index == self._size:
            self.extend(values)
            return

        index = self._format_int_index(index)

        new_size = self._size + len(values)
        if new_size > self._cap:
            self._grow_data(new_size)

        self._data[index + len(values) : new_size] = self._data[index : self._size]
        self._data[index : index + len(values)] = values
        self._size = new_size

    def remove(self, value, remove_all=False, from_right=False) -> bool:
        """
        Remove the first or all occurrences of a value from the vector.

        Parameters:
            value: The value to be removed.
            remove_all (bool): If True, remove all occurrences of value. Default is False.
            from_right (bool): If True, remove the rightmost occurrence. Default is False.

        Returns:
            bool: True if at least one occurrence was removed, False otherwise.
        """
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
        """
        Remove and return the item at the given index. If no index is specified, removes and returns the last item.

        Parameters:
            index (int): The position of the item to be removed. Defaults to -1 (the last item).
            return_None (bool): If True, return None instead of raising an IndexError when popping from an empty vector.

        Returns:
            The value at the specified index, or None if return_None is True and the vector is empty.

        Notes:
            1) Removing the last item has no performance penalty as it doesn't require shifting array elements.
            2) Popping from an empty vector with `return_None=False` raises an IndexError.
        """
        if isinstance(index, (int, np.integer, np.unsignedinteger)):
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
        """
        Remove the element at the given index or indices.

        If a list-like set of indices are provided, the elements are removed in
        descending order (largest index first), ensuring that all the requested
        indices are removed correctly.

        Parameters:
            index (int or list-like): The index or indices of the elements to remove. Defaults to -1 (last element).

        Notes:
            1) Removing the last item has no performance penalty as it doesn't require shifting array elements.
        """
        if self._size < 1:
            return

        if isinstance(index, (int, np.integer, np.unsignedinteger)):
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
        """
        Count the number of occurrences of a value in the vector.

        Parameters:
            value: The value to count.

        Returns:
            int: The number of occurrences of the value.
        """
        return len(self.where(value))

    def sort(self, reverse=False, kind=None):
        """
        Sort the vector in place in ascending order by default.

        Parameters:
            reverse (bool): If True, the vector is sorted in descending order. Defaults to False.
            kind (str): The sorting algorithm to use. Options are 'quicksort', 'heapsort', and 'stable'. Defaults to 'quicksort'.
        """
        if self._size < 1:
            return
        self._data[: self._size].sort(kind=kind)

        if reverse:
            self._reverse_in_place(self._data[: self._size])

    def reverse(self):
        """
        Reverse the order of elements in the vector.
        """
        self._reverse_in_place(self._data[: self._size])

    def contains(self, value) -> bool:
        """
        Check if the vector contains the specified value.
        `self.contains(value)` is equivalent to `value in self`

        Parameters:
            value: The value to check for.

        Returns:
            bool: True if the value is found in the vector, False otherwise.
        """
        return value in self._data[: self._size]

    def copy(self, min_capacity=8):
        """
        Create a copy of the current vector.

        Parameters:
            min_capacity (int): The minimum capacity of the copied vector. Defaults to 8.

        Returns:
            DynamicVector: A new vector that is a copy of the current vector.
        """
        capacity = self._size if min_capacity < self._size else min_capacity
        dyn = DynamicVector(self._dtype, capacity, grow_use_add=self._grow_use_add, grow_add=self._grow_add)
        dyn.extend(self._data[: self._size])
        return dyn

    def clear(self):
        """
        Clear the vector, removing all elements and resetting its size to zero.
        """
        self._size = 0

    def abs(self, where=True):
        """
        Compute the absolute value of the elements in the vector in place.

        Parameters:
            where (bool): A boolean mask specifying where to compute the absolute values. Default is True (apply to all elements).
        """
        vec = self._data[: self._size]
        np.absolute(vec, out=vec, where=where)

    def where(self, value):
        """
        Find the indices of all occurrences of the specified value in the vector.

        Parameters:
            value: The value to search for.

        Returns:
            np.ndarray: An array of indices where the value occurs in the vector.
        """
        return np.where(self._data[: self._size] == value)[0]

    def index(self, value, from_right=False) -> np.integer:
        p = np.where(self._data[: self._size] == value)[0]

        if len(p) < 1:
            raise ValueError(f"{value} is not in the vector")

        if from_right:
            return p[-1]
        return p[0]

    def resize(self, size: int):
        """
        Change the size of the vector.

        Note that resizing only changes the size of the vector
        and not the values stored in it. It is not recommended to
        rely on the values stored outside of the size of the vector
        as they will be overwritten with append and insert operations.


        The increased part of the vector has an undefined behavior until assigned a value.
        It will not raise an error, but the value stored in the new locations
        is either a previously stored value or zero.

        Parameters:
            size (int): The new size of the vector.

        Notes:
            1) Resizing has no performance penalty as long as it is less than the capacity.
               Otherwise, it will require a reallocation to increase the array capacity.
        """
        if size < 0:
            self.clear()
            return

        # if size < self._size:
        #     self._data[size : self._size] = self._zero
        if size > self._cap:
            self._grow_data(size)
        self._size = size

    def increase_size(self, increase_by: int):
        """
        Increase the size of the vector by a specified amount.
        The increased part of the vector has an undefined behavior until assigned a value.
        It will not raise an error, but the value stored in the new locations
        is either a previously stored value or zero.

        Parameters:
            increase_by (int): The number of elements to increase the vector's size by.

        Notes:
            1) Increasing the size has no performance penalty as long as it is less than the capacity.
               Otherwise, it will require a reallocation to increase the array capacity.

        """
        self.resize(self._size + increase_by)

    def set_capacity(self, min_capacity):
        """
        Ensures that the vector capacity is at least min_capacity.

        Note, if the capacity is changed, then it is set to
        the smallest power of two that exceeds min_capacity.

        Parameters:
            min_capacity (int): The minimum capacity the vector should support.
        """
        if min_capacity > self._cap:
            self._grow_data(min_capacity)

    def force_capacity(self, min_capacity):
        """
        Set the capacity of the vector to the smallest power of two
        that exceeds the specified minimum capacity.

        Parameters:
            min_capacity (int): The minimum capacity the vector should support.
        """
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

    def is_equal(self, other) -> bool:
        """
        Check if all elements in the vector are equal to the
        corresponding elements in another array or single value.
        If the array is a different size, then returns False.

        Parameters:
            other (value or array-like): The array to compare with.

        Returns:
            bool: True if all elements are equal, False otherwise.
        """
        try:
            if len(other) != self._size:
                return False
        except TypeError:
            pass
        return np.all(self.view == other)

    def is_less(self, other) -> bool:
        """
        Check if all elements in the vector are less than
        the corresponding elements in another array or single value.

        Parameters:
            other (value or array-like): The array to compare with.

        Returns:
            bool: True if all elements are less, False otherwise.

        Notes:
            1) If len(array-like) != len(self), then a TypeError is raised.
        """
        return np.all(self.view < other)

    def is_greater(self, other) -> bool:
        """
        Check if all elements in the vector are greater than
        the corresponding elements in another array or single value.

        Parameters:
            other (value or array-like): The array to compare with.

        Returns:
            bool: True if all elements are greater, False otherwise.

        Notes:
            1) If len(array-like) != len(self), then a TypeError is raised.
        """
        return np.all(self.view > other)

    def is_less_or_equal(self, other) -> bool:
        """
        Check if all elements in the vector are less than or equal to
        the corresponding elements in another array or single value.

        Parameters:
            other (value or array-like): The array to compare with.

        Returns:
            bool: True if all elements are less than or equal to, False otherwise.

        Notes:
            1) If len(array-like) != len(self), then a TypeError is raised.
        """
        return np.all(self.view <= other)

    def is_greater_or_equal(self, other) -> bool:
        """
        Check if all elements in the vector are greater than or equal to
        the corresponding elements in another array or single value.

        Parameters:
            other (value or array-like): The array to compare with.

        Returns:
            bool: True if all elements are greater than or equal to, False otherwise.

        Notes:
            1) If len(array-like) != len(self), then a TypeError is raised.
        """
        return np.all(self.view >= other)

    def is_not_equal(self, other) -> bool:
        """
        Check if all elements in the vector are NOT equal to the
        corresponding elements in another array or single value.
        If the array is a different size, then returns True.

        Parameters:
            other (value or array-like): The array to compare with.

        Returns:
            bool: True if all elements are equal, False otherwise.
        """
        try:
            if len(other) != self._size:
                return True
        except TypeError:
            pass
        return np.all(self.view != other)

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer, np.unsignedinteger)):
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
        if isinstance(index, (int, np.integer, np.unsignedinteger)):
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

    def __len__(self) -> int:
        return self._size

    def __repr__(self):
        dyn = repr(self.view)
        dyn = dyn[dyn.find("(") :]  # drop "array" from name
        return f"DynamicVector{dyn}"

    def __str__(self):
        dyn = repr(self.view)
        dyn = dyn[dyn.find("[") : dyn.find("]") + 1]  # get core part of numpy array
        return f"DynamicVector({dyn})"

    def __iter__(self):
        return iter(self.view)

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, unused=None):
        return self.copy()

    def __contains__(self, value):
        return value in self._data[: self._size]

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
        self._data[: self._size] += other
        return self

    def __isub__(self, other):  # To get called on subtraction with assignment e.g. a -=b.
        if isinstance(other, DynamicVector):
            other = other.view
        self._data[: self._size] -= other
        return self

    def __imul__(self, other):  # To get called on multiplication with assignment e.g. a *=b.
        if isinstance(other, DynamicVector):
            other = other.view
        self._data[: self._size] *= other
        return self

    def __itruediv__(self, other):  # To get called on true division with assignment e.g. a /=b.
        if isinstance(other, DynamicVector):
            other = other.view
        self._data[: self._size] /= other
        return self

    def __irtruediv__(self, other):  # To get called on true division with assignment e.g. a /=b.
        if isinstance(other, DynamicVector):
            other = other.view
        self._data[: self._size] /= other
        return self

    def __ifloordiv__(self, other):  # To get called on integer division with assignment e.g. a //=b.
        if isinstance(other, DynamicVector):
            other = other.view
        self._data[: self._size] //= other
        return self

    def __irfloordiv__(self, other):  # To get called on integer division with assignment e.g. a //=b.
        if isinstance(other, DynamicVector):
            other = other.view
        self._data[: self._size] //= other
        return self

    def __imod__(self, other):  # To get called on modulo with assignment e.g. a%=b.
        if isinstance(other, DynamicVector):
            other = other.view
        self._data[: self._size] %= other
        return self

    def __irmod__(self, other):  # To get called on modulo with assignment e.g. a%=b.
        if isinstance(other, DynamicVector):
            other = other.view
        self._data[: self._size] %= other
        return self

    def __ipow__(self, other):  # To get called on exponents with assignment e.g. a **=b.
        if isinstance(other, DynamicVector):
            other = other.view
        self._data[: self._size] **= other
        return self

    def __irpow__(self, other):  # To get called on exponents with assignment e.g. a **=b.
        if isinstance(other, DynamicVector):
            other = other.view
        self._data[: self._size] **= other
        return self

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
        """
        Convert a slice object to a range of indices.

        Parameters:
            sl (slice): The slice to convert.
            same_order (bool): If True, preserve the order of the slice. Defaults to False (descending order).

        Returns:
            range: A range of indices corresponding to the slice.
        """
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
        """
        Increase the internal storage capacity of the vector
        to accommodate at least the specified minimum capacity.

        Parameters:
            min_capacity (int): The minimum capacity the vector should have after growth.
        """
        self._setup_capacity(min_capacity)
        old_cap = len(self._data)
        if old_cap < self._cap:  # should always be true
            tmp = np.zeros(self._cap, dtype=self._dtype)
            tmp[:old_cap] = self._data
            self._data = tmp
        else:
            self._cap = old_cap

    def _setup_capacity(self, min_capacity):
        """
        Adjust the vector's internal capacity variable (self._cap)
        to be a power of two or incrementally increase it
        if necessary, to meet or exceed the specified minimum capacity.

        Parameters:
            min_capacity (int): The minimum capacity the vector must support.
        """
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

    x = DynamicVector.from_values([1, 2, 3, 4, 5])

    # Initialize with a list
    vec = DynamicVector.from_values([1, 2, 3])  # integer vector with three values

    print(vec)
    print(repr(vec))
    # Output: DynamicVector([1, 2, 3])
    # Output: DynamicVector([1, 2, 3], dtype=int32)

    # Access the underlying NumPy array via the 'view' property
    print(vec.view)
    print(vec[:])
    # Output: [1 2 3]
    # Output: [1 2 3] -> same as vec.view

    # Perform NumPy operations
    vec += 1
    print(vec)
    # Output: DynamicVector([2, 3, 4])

    vec[1] = 99
    print(vec)
    # Output: DynamicVector([ 2, 99,  4])

    vec[1 : len(vec)] = 8  # set element 2 and 3 to the value of 8.
    print(vec)
    # Output: DynamicVector([2, 8, 8])

    vec[:] = [2, 3, 4]
    print(vec)
    # Output: DynamicVector([2, 3, 4])

    # Append elements dynamically
    vec.append(5)  # Fast operation
    print(vec)
    # Output: DynamicVector([2, 3, 4, 5])

    vec.extend([7, 8, 9])  # Fast operation
    print(vec)
    # Output: DynamicVector([1, 2, 3, 5, 7, 8, 9])

    # Insert at a specific index
    vec.insert(1, 10)
    print(vec)
    # Output: DynamicVector([ 1, 10,  2,  3,  5,  7,  8,  9])

    # Insert at a specific index
    vec.insert_values(3, [97, 98])
    print(vec)
    # Output: DynamicVector([ 1, 10,  2, 97, 98,  3,  5,  7,  8,  9])

    # Remove and return the last element
    print(vec)
    last_elem = vec.pop()  # Fast operation
    print(vec)
    print(last_elem)
    # Output: DynamicVector([ 1, 10,  2, 97, 98,  3,  5,  7,  8,  9])
    # Output: DynamicVector([ 1, 10,  2, 97, 98,  3,  5,  7,  8])
    # Output: 9

    third_element = vec.pop(2)
    print(vec)
    print(third_element)
    # Output: DynamicVector([ 1, 10, 97, 98,  3,  5,  7,  8])
    # Output: 2

    # Slice behaves like NumPy arrays
    sliced_vec = vec[1:3]
    print(sliced_vec)
    # Output: [10  97]

    vec[2:5] = [51, 52, 53]
    print(vec)
    # Output: DynamicVector([ 1, 10, 51, 52, 53,  5,  7,  8])

    vec[[1, 3, 5]] = [-1, -2, -3]
    print(vec)
    # Output: DynamicVector([ 1, -1, 51, -2, 53, -3,  7,  8])
