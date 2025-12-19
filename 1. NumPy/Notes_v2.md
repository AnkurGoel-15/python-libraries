# NumPy Complete Guide for Data Engineers & Analysts

## Table of Contents
1. [Introduction](#introduction)
2. [Installation & Setup](#installation--setup)
3. [Array Creation](#array-creation)
4. [Array Attributes](#array-attributes)
5. [Array Manipulation](#array-manipulation)
6. [Indexing & Slicing](#indexing--slicing)
7. [Mathematical Operations](#mathematical-operations)
8. [Statistical Operations](#statistical-operations)
9. [Linear Algebra](#linear-algebra)
10. [Random Numbers](#random-numbers)
11. [Array Comparison & Logic](#array-comparison--logic)
12. [Broadcasting](#broadcasting)
13. [File I/O](#file-io)
14. [Performance Tips](#performance-tips)
15. [Quick Reference](#quick-reference)

---

## Introduction

**NumPy (Numerical Python)** is the fundamental package for scientific computing in Python. It provides:
- Fast and efficient multidimensional array object (`ndarray`)
- Mathematical functions for array operations
- Tools for working with linear algebra, Fourier transforms, and random numbers
- Foundation for other libraries like Pandas, SciPy, and Scikit-learn

### Why NumPy?

- **Speed**: NumPy arrays are up to 50x faster than Python lists
- **Memory Efficient**: Uses less memory than Python lists
- **Vectorization**: Perform operations on entire arrays without loops
- **Broadcasting**: Automatic element-wise operations on arrays of different shapes
- **Integration**: Works seamlessly with other scientific Python libraries

### Applications

1. **Data Science & Machine Learning**: Feature engineering, data preprocessing
2. **Scientific Computing**: Numerical simulations, signal processing
3. **Image Processing**: Manipulation of image arrays
4. **Financial Analysis**: Time series analysis, portfolio optimization

---

## Installation & Setup

```bash
# Install NumPy
pip install numpy

# Install specific version
pip install numpy==1.24.0
```

### **Import NumPy**

```python
import numpy as np
```

- **np.\_\_version\_\_**: Check NumPy version

**Example:**
```python
import numpy as np
print(np.__version__)
```

**Output:**
```
1.24.3
```

---

## Array Creation

### **Basic Array Creation**

- **np.array(list)**: Create array from Python list or tuple
- **np.array(list, dtype=type)**: Create array with specific data type
- **np.asarray(data)**: Convert input to array (won't copy if already array)
- **np.copy(array)**: Create a deep copy of array

**Example:**
```python
import numpy as np

# 1D array from list
a = np.array([56, 78, 90, 65, 88, 73, 94])
print(a)
```

**Output:**
```
[56 78 90 65 88 73 94]
```

```python
# 2D array
a_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(a_2d)
```

**Output:**
```
[[1 2 3]
 [4 5 6]]
```

```python
# Array with specific dtype
arr = np.array([1, 2, 3], dtype=float)
print(arr)
```

**Output:**
```
[1. 2. 3.]
```

### **Arrays with Default Values**

- **np.zeros(shape)**: Create array filled with zeros
- **np.zeros(shape, dtype=type)**: Create zeros array with specific type
- **np.ones(shape)**: Create array filled with ones
- **np.full(shape, value)**: Create array filled with specific value
- **np.empty(shape)**: Create uninitialized array (faster, but contains garbage values)
- **np.zeros_like(array)**: Create zeros array with same shape as given array
- **np.ones_like(array)**: Create ones array with same shape as given array
- **np.full_like(array, value)**: Create filled array with same shape

**Example:**
```python
# Zeros array
b = np.zeros((2, 3))
print(b)
```

**Output:**
```
[[0. 0. 0.]
 [0. 0. 0.]]
```

```python
# Ones array
c = np.ones((2, 3))
print(c)
```

**Output:**
```
[[1. 1. 1.]
 [1. 1. 1.]]
```

```python
# Array filled with specific value
d = np.full((3, 3), 7)
print(d)
```

**Output:**
```
[[7 7 7]
 [7 7 7]
 [7 7 7]]
```

### **Identity and Diagonal Arrays**

- **np.eye(n)**: Create n×n identity matrix
- **np.eye(n, m)**: Create n×m identity matrix
- **np.identity(n)**: Create n×n identity matrix (same as np.eye)
- **np.diag(array)**: Create diagonal matrix from 1D array
- **np.diag(matrix)**: Extract diagonal from 2D array

**Example:**
```python
# Identity matrix
e = np.eye(3)
print(e)
```

**Output:**
```
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

```python
# Diagonal matrix
diag = np.diag([1, 2, 3])
print(diag)
```

**Output:**
```
[[1 0 0]
 [0 2 0]
 [0 0 3]]
```

### **Range Arrays**

- **np.arange(start, stop, step)**: Create array with range of values (like Python range)
- **np.arange(stop)**: Create array from 0 to stop
- **np.linspace(start, stop, num)**: Create array with num evenly spaced values
- **np.logspace(start, stop, num)**: Create array with logarithmically spaced values
- **np.geomspace(start, stop, num)**: Create array with geometrically spaced values

**Example:**
```python
# Array with range
f = np.arange(0, 10, 2)
print(f)
```

**Output:**
```
[0 2 4 6 8]
```

```python
# Evenly spaced values
g = np.linspace(0, 1, 5)
print(g)
```

**Output:**
```
[0.   0.25 0.5  0.75 1.  ]
```

```python
# Logarithmically spaced
log_arr = np.logspace(0, 2, 5)  # 10^0 to 10^2
print(log_arr)
```

**Output:**
```
[  1.           3.16227766  10.          31.6227766  100.        ]
```

### **Random Arrays**

- **np.random.rand(shape)**: Random floats in [0, 1) from uniform distribution
- **np.random.randn(shape)**: Random floats from standard normal distribution
- **np.random.randint(low, high, size)**: Random integers in [low, high)
- **np.random.random(size)**: Random floats in [0, 1)
- **np.random.choice(array, size)**: Random samples from given array
- **np.random.seed(seed)**: Set random seed for reproducibility

**Example:**
```python
# Random floats in [0, 1)
h = np.random.rand(2, 2)
print(h)
```

**Output:**
```
[[0.32603611 0.37116063]
 [0.32458082 0.84352092]]
```

```python
# Random integers
i = np.random.randint(0, 10, (2, 3))
print(i)
```

**Output:**
```
[[5 4 2]
 [0 5 1]]
```

```python
# Random from normal distribution
normal = np.random.randn(3, 3)
print(normal)
```

**Output:**
```
[[-0.12345678  0.98765432 -1.23456789]
 [ 0.45678901 -0.56789012  0.67890123]
 [-0.78901234  0.89012345 -0.90123456]]
```

```python
# Set seed for reproducibility
np.random.seed(42)
arr1 = np.random.rand(3)
np.random.seed(42)
arr2 = np.random.rand(3)
print(arr1 == arr2)  # All True
```

---

## Array Attributes

### **Shape and Size**

- **array.shape**: Get dimensions of array (returns tuple)
- **array.size**: Get total number of elements
- **array.ndim**: Get number of dimensions
- **array.dtype**: Get data type of elements
- **array.itemsize**: Get size in bytes of each element
- **array.nbytes**: Get total bytes consumed by array elements

**Example:**
```python
import numpy as np

marks = np.array([
    [85, 78, 92],
    [66, 74, 81],
    [90, 88, 95],
    [70, 65, 60],
])

print(marks.shape)
```

**Output:**
```
(4, 3)
```

```python
print(marks.size)
```

**Output:**
```
12
```

```python
print(marks.ndim)
```

**Output:**
```
2
```

```python
print(marks.dtype)
```

**Output:**
```
int64
```

```python
print(marks.itemsize)
```

**Output:**
```
8
```

```python
print(marks.nbytes)
```

**Output:**
```
96
```

### **Data Types**

- **array.astype(dtype)**: Convert array to different data type
- **np.int32, np.int64**: Integer types
- **np.float32, np.float64**: Float types
- **np.bool_**: Boolean type
- **np.complex64, np.complex128**: Complex number types
- **np.str_**: String type

**Example:**
```python
# Convert to float
float_marks = marks.astype(float)
print(float_marks.dtype)
```

**Output:**
```
float64
```

```python
# Convert to int32 (saves memory)
int32_marks = marks.astype(np.int32)
print(int32_marks.dtype)
```

**Output:**
```
int32
```

---

## Array Manipulation

### **Reshaping Arrays**

- **array.reshape(new_shape)**: Change shape without changing data
- **array.resize(new_shape)**: Change shape and modify array in-place
- **array.ravel()**: Flatten array to 1D (returns view if possible)
- **array.flatten()**: Flatten array to 1D (always returns copy)
- **array.squeeze()**: Remove single-dimensional entries
- **np.expand_dims(array, axis)**: Add new dimension at specified axis

**Example:**
```python
import numpy as np

# Create and reshape array
a = np.arange(6).reshape(2, 3)
print(a)
```

**Output:**
```
[[0 1 2]
 [3 4 5]]
```

```python
# Reshape to different dimensions
b = np.reshape(a, (3, 2))
print(b)
```

**Output:**
```
[[0 1]
 [2 3]
 [4 5]]
```

```python
# Flatten to 1D
c = np.ravel(a)
print(c)
```

**Output:**
```
[0 1 2 3 4 5]
```

```python
# Reshape to -1 (automatic dimension calculation)
auto_reshape = np.arange(12).reshape(3, -1)  # -1 means "calculate this dimension"
print(auto_reshape)
```

**Output:**
```
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]
```

### **Transposing Arrays**

- **array.T**: Transpose array (swap axes)
- **np.transpose(array)**: Transpose array
- **np.transpose(array, axes)**: Permute array dimensions
- **array.swapaxes(axis1, axis2)**: Swap two axes

**Example:**
```python
# Transpose
d = np.transpose(a)
print(d)
```

**Output:**
```
[[0 3]
 [1 4]
 [2 5]]
```

```python
# Using .T
print(a.T)
```

**Output:**
```
[[0 3]
 [1 4]
 [2 5]]
```

### **Joining Arrays**

- **np.concatenate([arr1, arr2], axis)**: Join arrays along existing axis
- **np.vstack([arr1, arr2])**: Stack arrays vertically (row-wise)
- **np.hstack([arr1, arr2])**: Stack arrays horizontally (column-wise)
- **np.dstack([arr1, arr2])**: Stack arrays depth-wise (along 3rd axis)
- **np.column_stack([arr1, arr2])**: Stack 1D arrays as columns
- **np.row_stack([arr1, arr2])**: Stack arrays as rows (same as vstack)

**Example:**
```python
# Concatenate along axis 0 (rows)
e = np.concatenate([a, a], axis=0)
print(e)
```

**Output:**
```
[[0 1 2]
 [3 4 5]
 [0 1 2]
 [3 4 5]]
```

```python
# Concatenate along axis 1 (columns)
f = np.concatenate([a, a], axis=1)
print(f)
```

**Output:**
```
[[0 1 2 0 1 2]
 [3 4 5 3 4 5]]
```

```python
# Vertical stack
g = np.vstack([a, a])
print(g)
```

**Output:**
```
[[0 1 2]
 [3 4 5]
 [0 1 2]
 [3 4 5]]
```

```python
# Horizontal stack
h = np.hstack([a, a])
print(h)
```

**Output:**
```
[[0 1 2 0 1 2]
 [3 4 5 3 4 5]]
```

### **Splitting Arrays**

- **np.split(array, sections, axis)**: Split array into multiple sub-arrays
- **np.array_split(array, sections, axis)**: Split allowing unequal division
- **np.hsplit(array, sections)**: Split horizontally (column-wise)
- **np.vsplit(array, sections)**: Split vertically (row-wise)
- **np.dsplit(array, sections)**: Split depth-wise (along 3rd axis)

**Example:**
```python
# Split into 2 parts
i = np.split(a, 2)
print(i)
```

**Output:**
```
[array([[0, 1, 2]]), array([[3, 4, 5]])]
```

```python
# Horizontal split
hsplit_arr = np.hsplit(a, 3)  # Split into 3 columns
print(hsplit_arr)
```

**Output:**
```
[array([[0],
        [3]]), array([[1],
        [4]]), array([[2],
        [5]])]
```

### **Adding and Removing Elements**

- **np.append(array, values, axis)**: Append values to end of array
- **np.insert(array, index, values, axis)**: Insert values at given index
- **np.delete(array, index, axis)**: Delete values at given index
- **np.unique(array)**: Find unique elements
- **np.unique(array, return_counts=True)**: Find unique elements with counts

**Example:**
```python
# Insert at index 0
j = np.insert(np.arange(6), 0, 100)
print(j)
```

**Output:**
```
[100   0   1   2   3   4   5]
```

```python
# Delete at index 0
k = np.delete(np.arange(6), 0)
print(k)
```

**Output:**
```
[1 2 3 4 5]
```

```python
# Append arrays
l = np.append(np.arange(3), np.arange(4))
print(l)
```

**Output:**
```
[0 1 2 0 1 2 3]
```

```python
# Find unique elements
unique_vals = np.unique([1, 2, 2, 3, 3, 3])
print(unique_vals)
```

**Output:**
```
[1 2 3]
```

---

## Indexing & Slicing

### **Basic Indexing**

- **array[index]**: Access element at index
- **array[row, col]**: Access element in 2D array
- **array[i, j, k]**: Access element in 3D array
- **array[-1]**: Access last element
- **array[-2]**: Access second-to-last element

**Example:**
```python
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

# Access element at row 0, column 1
print(a[0, 1])
```

**Output:**
```
2
```

```python
# Access last element
print(a[-1, -1])
```

**Output:**
```
6
```

### **Slicing**

- **array[start:stop]**: Slice 1D array
- **array[start:stop:step]**: Slice with step
- **array[:, col]**: Select entire column
- **array[row, :]**: Select entire row
- **array[start:stop, start:stop]**: Slice 2D array

**Example:**
```python
# Select entire column 2
print(a[:, 2])
```

**Output:**
```
[3 6]
```

```python
# Select first row
print(a[0, :])
```

**Output:**
```
[1 2 3]
```

```python
# Slice 2D array
print(a[0:2, 1:3])
```

**Output:**
```
[[2 3]
 [5 6]]
```

### **Boolean Indexing**

- **array[condition]**: Select elements that satisfy condition
- **array[array > value]**: Select elements greater than value
- **array[(cond1) & (cond2)]**: Multiple conditions with AND
- **array[(cond1) | (cond2)]**: Multiple conditions with OR
- **array[~condition]**: NOT condition

**Example:**
```python
# Boolean indexing
print(a[a > 3])
```

**Output:**
```
[4 5 6]
```

```python
# Multiple conditions
print(a[(a > 2) & (a < 6)])
```

**Output:**
```
[3 4 5]
```

### **Fancy Indexing**

- **array[[indices]]**: Select elements at specific indices
- **array[[rows], [cols]]**: Select specific elements by row and column indices
- **array[np.ix_([rows], [cols])]**: Select subarray using index arrays

**Example:**
```python
# Fancy indexing
print(a[[0, 1], [1, 2]])
```

**Output:**
```
[2 6]
```

```python
# Using np.ix_
rows = [0, 1]
cols = [1, 2]
print(a[np.ix_(rows, cols)])
```

**Output:**
```
[[2 3]
 [5 6]]
```

---

## Mathematical Operations

### **Element-wise Operations**

- **array + value**: Add scalar to all elements
- **array - value**: Subtract scalar from all elements
- **array * value**: Multiply all elements by scalar
- **array / value**: Divide all elements by scalar
- **array ** power**: Raise all elements to power
- **array % value**: Modulo operation
- **array // value**: Floor division

**Example:**
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Add scalar
print(a + 5)
```

**Output:**
```
[6 7 8]
```

```python
# Power
print(a ** 2)
```

**Output:**
```
[1 4 9]
```

```python
# Element-wise addition
print(a + b)
```

**Output:**
```
[5 7 9]
```

```python
# Element-wise multiplication
print(a * b)
```

**Output:**
```
[ 4 10 18]
```

### **Mathematical Functions**

- **np.add(arr1, arr2)**: Add arrays element-wise
- **np.subtract(arr1, arr2)**: Subtract arrays element-wise
- **np.multiply(arr1, arr2)**: Multiply arrays element-wise
- **np.divide(arr1, arr2)**: Divide arrays element-wise
- **np.power(array, exponent)**: Raise to power element-wise
- **np.mod(arr1, arr2)**: Modulo element-wise
- **np.abs(array)**: Absolute value
- **np.sign(array)**: Sign of elements (-1, 0, 1)

**Example:**
```python
# Using np functions
print(np.add(a, b))
print(np.multiply(a, b))
print(np.power(a, 2))
```

### **Exponential and Logarithmic**

- **np.exp(array)**: e^x for each element
- **np.exp2(array)**: 2^x for each element
- **np.log(array)**: Natural logarithm
- **np.log10(array)**: Base 10 logarithm
- **np.log2(array)**: Base 2 logarithm
- **np.sqrt(array)**: Square root
- **np.cbrt(array)**: Cube root

**Example:**
```python
# Exponential
print(np.exp(a))
```

**Output:**
```
[ 2.71828183  7.3890561  20.08553692]
```

```python
# Square root
print(np.sqrt(a))
```

**Output:**
```
[1.         1.41421356 1.73205081]
```

```python
# Natural logarithm
print(np.log(a))
```

**Output:**
```
[0.         0.69314718 1.09861229]
```

### **Trigonometric Functions**

- **np.sin(array)**: Sine
- **np.cos(array)**: Cosine
- **np.tan(array)**: Tangent
- **np.arcsin(array)**: Inverse sine
- **np.arccos(array)**: Inverse cosine
- **np.arctan(array)**: Inverse tangent
- **np.deg2rad(array)**: Convert degrees to radians
- **np.rad2deg(array)**: Convert radians to degrees

**Example:**
```python
# Trigonometric functions
angles = np.array([0, 30, 45, 60, 90])
radians = np.deg2rad(angles)
print(np.sin(radians))
```

**Output:**
```
[0.         0.5        0.70710678 0.8660254  1.        ]
```

### **Rounding**

- **np.round(array, decimals)**: Round to given number of decimals
- **np.floor(array)**: Round down to nearest integer
- **np.ceil(array)**: Round up to nearest integer
- **np.trunc(array)**: Truncate to integer (remove decimal part)
- **np.rint(array)**: Round to nearest integer

**Example:**
```python
values = np.array([1.2, 2.5, 3.8, 4.1])

print(np.round(values))    # [1. 2. 4. 4.]
print(np.floor(values))    # [1. 2. 3. 4.]
print(np.ceil(values))     # [2. 3. 4. 5.]
print(np.trunc(values))    # [1. 2. 3. 4.]
```

---

## Statistical Operations

### **Basic Statistics**

- **np.sum(array)**: Sum of all elements
- **np.sum(array, axis)**: Sum along specified axis
- **np.mean(array)**: Mean (average) of elements
- **np.median(array)**: Median of elements
- **np.std(array)**: Standard deviation
- **np.var(array)**: Variance
- **np.min(array)**: Minimum value
- **np.max(array)**: Maximum value
- **np.ptp(array)**: Peak-to-peak (max - min)

**Example:**
```python
import numpy as np

a = np.array([1, 2, 3])

# Sum
print(np.sum(a))
```

**Output:**
```
6
```

```python
# Mean
print(np.mean(a))
```

**Output:**
```
2.0
```

```python
# Standard deviation
print(np.std(a))
```

**Output:**
```
0.816496580927726
```

```python
# Min and Max
print(np.min(a), np.max(a))
```

**Output:**
```
1 3
```

### **Advanced Statistics**

- **np.percentile(array, q)**: q-th percentile
- **np.quantile(array, q)**: q-th quantile
- **np.average(array, weights)**: Weighted average
- **np.corrcoef(x, y)**: Correlation coefficient
- **np.cov(x, y)**: Covariance matrix
- **np.histogram(array, bins)**: Compute histogram

**Example:**
```python
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Percentiles
print(np.percentile(data, 50))  # Median
```

**Output:**
```
5.5
```

```python
# Histogram
hist, bins = np.histogram(data, bins=5)
print(hist)
print(bins)
```

**Output:**
```
[2 2 2 2 2]
[ 1.   2.8  4.6  6.4  8.2 10. ]
```

### **Cumulative Operations**

- **np.cumsum(array)**: Cumulative sum
- **np.cumprod(array)**: Cumulative product
- **np.nancumsum(array)**: Cumulative sum ignoring NaN
- **np.nancumprod(array)**: Cumulative product ignoring NaN

**Example:**
```python
arr = np.array([1, 2, 3, 4])

print(np.cumsum(arr))    # [ 1  3  6 10]
print(np.cumprod(arr))   # [ 1  2  6 24]
```

### **Argument Functions**

- **np.argmin(array)**: Index of minimum value
- **np.argmax(array)**: Index of maximum value
- **np.argsort(array)**: Indices that would sort array
- **np.argwhere(condition)**: Indices where condition is True
- **np.nonzero(array)**: Indices of non-zero elements

**Example:**
```python
arr = np.array([3, 1, 2, 5, 4])

print(np.argmin(arr))    # 1 (index of minimum)
print(np.argmax(arr))    # 3 (index of maximum)
print(np.argsort(arr))   # [1 2 0 4 3] (indices for sorted array)
```

---

## Linear Algebra

### **Matrix Operations**

- **np.dot(a, b)**: Dot product of two arrays
- **np.matmul(a, b)**: Matrix multiplication
- **a @ b**: Matrix multiplication operator (Python 3.5+)
- **np.inner(a, b)**: Inner product
- **np.outer(a, b)**: Outer product
- **np.cross(a, b)**: Cross product

**Example:**
```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Dot product
print(np.dot(a, b))
```

**Output:**
```
32
```

```python
# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.matmul(A, B))
```

**Output:**
```
[[19 22]
 [43 50]]
```

```python
# Using @ operator
print(A @ B)
```

**Output:**
```
[[19 22]
 [43 50]]
```

### **Matrix Decomposition**

- **np.linalg.det(matrix)**: Determinant of matrix
- **np.linalg.inv(matrix)**: Inverse of matrix
- **np.linalg.eig(matrix)**: Eigenvalues and eigenvectors
- **np.linalg.svd(matrix)**: Singular Value Decomposition
- **np.linalg.qr(matrix)**: QR decomposition
- **np.linalg.cholesky(matrix)**: Cholesky decomposition

**Example:**
```python
matrix = np.array([[1, 2], [3, 4]])

# Determinant
print(np.linalg.det(matrix))
```

**Output:**
```
-2.0
```

```python
# Inverse
print(np.linalg.inv(matrix))
```

**Output:**
```
[[-2.   1. ]
 [ 1.5 -0.5]]
```

```python
# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print(eigenvalues)
```

**Output:**
```
[-0.37228132  5.37228132]
```

### **Solving Linear Systems**

- **np.linalg.solve(A, b)**: Solve linear system Ax = b
- **np.linalg.lstsq(A, b)**: Least-squares solution
- **np.linalg.norm(array)**: Matrix or vector norm
- **np.trace(matrix)**: Trace of matrix (sum of diagonal)
- **np.linalg.matrix_rank(matrix)**: Rank of matrix

**Example:**
```python
# Solve Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

x = np.linalg.solve(A, b)
print(x)
```

**Output:**
```
[2. 3.]
```

```python
# Verify solution
print(np.dot(A, x))  # Should equal b
```

**Output:**
```
[9. 8.]
```

---

## Random Numbers

### **Random Generation**

- **np.random.seed(seed)**: Set random seed for reproducibility
- **np.random.rand(d0, d1, ...)**: Random floats in [0, 1) uniform distribution
- **np.random.randn(d0, d1, ...)**: Random floats from standard normal distribution
- **np.random.randint(low, high, size)**: Random integers
- **np.random.random(size)**: Random floats in [0, 1)
- **np.random.uniform(low, high, size)**: Random floats from uniform distribution
- **np.random.normal(mean, std, size)**: Random floats from normal distribution
- **np.random.choice(array, size, replace)**: Random samples from array
- **np.random.shuffle(array)**: Shuffle array in-place
- **np.random.permutation(array)**: Return shuffled copy

**Example:**
```python
# Set seed for reproducibility
np.random.seed(42)

# Random uniform
uniform = np.random.uniform(0, 10, 5)
print(uniform)
```

**Output:**
```
[3.74540119 9.50714306 7.31993942 5.98658484 1.5601864 ]
```

```python
# Random normal distribution
normal = np.random.normal(0, 1, 5)  # mean=0, std=1
print(normal)
```

**Output:**
```
[ 0.37454012 -0.15601864  0.95071431  0.73199394 -0.59865848]
```

```python
# Random choice
choices = np.random.choice([1, 2, 3, 4, 5], size=3, replace=False)
print(choices)
```

**Output:**
```
[3 1 4]
```

### **Random Distributions**

- **np.random.binomial(n, p, size)**: Binomial distribution
- **np.random.poisson(lam, size)**: Poisson distribution
- **np.random.exponential(scale, size)**: Exponential distribution
- **np.random.gamma(shape, scale, size)**: Gamma distribution
- **np.random.beta(a, b, size)**: Beta distribution
- **np.random.chisquare(df, size)**: Chi-square distribution

**Example:**
```python
# Binomial distribution (n trials, p probability)
binomial = np.random.binomial(10, 0.5, 1000)
print(np.mean(binomial))  # Should be close to n*p = 5
```

**Output:**
```
4.987
```

---

## Array Comparison & Logic

### **Comparison Operations**

- **array == value**: Element-wise equality
- **array != value**: Element-wise inequality
- **array > value**: Element-wise greater than
- **array < value**: Element-wise less than
- **array >= value**: Element-wise greater than or equal
- **array <= value**: Element-wise less than or equal

**Example:**
```python
arr = np.array([1, 2, 3, 4, 5])

print(arr > 3)
```

**Output:**
```
[False False False  True  True]
```

```python
print(arr == 3)
```

**Output:**
```
[False False  True False False]
```

### **Logical Operations**

- **np.logical_and(arr1, arr2)**: Element-wise AND
- **np.logical_or(arr1, arr2)**: Element-wise OR
- **np.logical_not(array)**: Element-wise NOT
- **np.logical_xor(arr1, arr2)**: Element-wise XOR
- **np.all(array)**: True if all elements are True
- **np.any(array)**: True if any element is True

**Example:**
```python
a = np.array([True, False, True])
b = np.array([True, True, False])

print(np.logical_and(a, b))
```

**Output:**
```
[ True False False]
```

```python
# Check if all elements satisfy condition
arr = np.array([2, 4, 6, 8])
print(np.all(arr > 0))  # True
print(np.any(arr > 5))  # True
```

### **Array Comparison Functions**

- **np.array_equal(arr1, arr2)**: True if arrays have same shape and elements
- **np.allclose(arr1, arr2, rtol, atol)**: True if arrays are element-wise equal within tolerance
- **np.isclose(arr1, arr2)**: Element-wise comparison within tolerance
- **np.greater(arr1, arr2)**: Element-wise greater than comparison
- **np.less(arr1, arr2)**: Element-wise less than comparison

**Example:**
```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([1.0, 2.0, 3.0001])

print(np.array_equal(a, b))        # False
print(np.allclose(a, b, rtol=1e-3))  # True (within tolerance)
```

### **Condition-based Selection**

- **np.where(condition, x, y)**: Return x where condition is True, else y
- **np.select(condlist, choicelist)**: Return elements from choicelist based on conditions
- **np.clip(array, min, max)**: Clip values to range [min, max]

**Example:**
```python
arr = np.array([1, 2, 3, 4, 5])

# Replace values > 3 with 0, else keep original
result = np.where(arr > 3, 0, arr)
print(result)
```

**Output:**
```
[1 2 3 0 0]
```

```python
# Clip values to range [2, 4]
clipped = np.clip(arr, 2, 4)
print(clipped)
```

**Output:**
```
[2 2 3 4 4]
```

---

## Broadcasting

**Broadcasting** allows NumPy to perform operations on arrays of different shapes efficiently.

### **Broadcasting Rules**

1. If arrays have different dimensions, pad smaller shape with ones on the left
2. Arrays are compatible if dimensions are equal or one of them is 1
3. Arrays are broadcast together to the larger shape

### **Broadcasting Examples**

- **scalar + array**: Broadcast scalar to array shape
- **1D array + 2D array**: Broadcast along compatible dimensions
- **Different shapes**: Arrays broadcast to common shape if compatible

**Example:**
```python
# Scalar broadcasting
arr = np.array([1, 2, 3])
print(arr + 10)
```

**Output:**
```
[11 12 13]
```

```python
# 1D + 2D broadcasting
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
row = np.array([10, 20, 30])

print(matrix + row)
```

**Output:**
```
[[11 22 33]
 [14 25 36]
 [17 28 39]]
```

```python
# Column broadcasting (need to reshape)
col = np.array([[10], [20], [30]])  # Shape (3, 1)
print(matrix + col)
```

**Output:**
```
[[11 12 13]
 [24 25 26]
 [37 38 39]]
```

```python
# Broadcasting visualization
a = np.array([[1], [2], [3]])  # Shape (3, 1)
b = np.array([10, 20, 30])      # Shape (3,)

print(a + b)  # Broadcasts to (3, 3)
```

**Output:**
```
[[11 21 31]
 [12 22 32]
 [13 23 33]]
```

---

## File I/O

### **Saving Arrays**

- **np.save('file.npy', array)**: Save single array to binary file
- **np.savez('file.npz', arr1=array1, arr2=array2)**: Save multiple arrays to compressed file
- **np.savez_compressed('file.npz', arr1=array1)**: Save with compression
- **np.savetxt('file.txt', array)**: Save array to text file
- **np.savetxt('file.csv', array, delimiter=',')**: Save as CSV

**Example:**
```python
arr = np.array([1, 2, 3, 4, 5])

# Save to binary file
np.save('my_array.npy', arr)

# Save multiple arrays
np.savez('multiple_arrays.npz', 
         first=arr, 
         second=arr*2)

# Save to text file
np.savetxt('array.txt', arr)

# Save to CSV
np.savetxt('array.csv', arr, delimiter=',')
```

### **Loading Arrays**

- **np.load('file.npy')**: Load array from binary file
- **np.load('file.npz')**: Load multiple arrays (returns dict-like object)
- **np.loadtxt('file.txt')**: Load array from text file
- **np.genfromtxt('file.csv', delimiter=',')**: Load from text with missing values handling

**Example:**
```python
# Load from binary file
loaded = np.load('my_array.npy')
print(loaded)

# Load multiple arrays
data = np.load('multiple_arrays.npz')
print(data['first'])
print(data['second'])

# Load from text file
loaded_txt = np.loadtxt('array.txt')
print(loaded_txt)

# Load CSV with specific delimiter
csv_data = np.genfromtxt('array.csv', delimiter=',')
print(csv_data)
```

---

## Performance Tips

### **Memory Optimization**

- **Use appropriate dtypes**: int32 instead of int64, float32 instead of float64
- **Use views instead of copies**: Slicing creates views by default
- **Use in-place operations**: Operations with `out` parameter
- **Avoid creating intermediate arrays**: Chain operations when possible
- **Use sparse arrays**: For arrays with many zeros

**Example:**
```python
# Use smaller dtypes
large_array = np.arange(1000000, dtype=np.int32)  # Half the memory of int64

# In-place operations
arr = np.array([1, 2, 3, 4, 5])
np.add(arr, 10, out=arr)  # Modify arr in-place

# Views vs copies
original = np.array([1, 2, 3, 4, 5])
view = original[1:4]      # This is a view
view[0] = 999             # Modifies original too
print(original)           # [1, 999, 3, 4, 5]

copy = original.copy()    # Explicit copy
copy[0] = 0               # Doesn't affect original
```

### **Vectorization**

**Always use vectorized operations instead of loops:**

```python
# ❌ BAD (Slow - using loop)
arr = np.arange(1000000)
result = np.zeros_like(arr)
for i in range(len(arr)):
    result[i] = arr[i] ** 2

# ✅ GOOD (Fast - vectorized)
arr = np.arange(1000000)
result = arr ** 2
```

### **Broadcasting vs Explicit Operations**

```python
# ✅ GOOD (Use broadcasting)
matrix = np.random.rand(1000, 1000)
row_mean = matrix.mean(axis=1, keepdims=True)
centered = matrix - row_mean  # Broadcasting

# ❌ BAD (Explicit loop)
centered = np.zeros_like(matrix)
for i in range(matrix.shape[0]):
    centered[i] = matrix[i] - matrix[i].mean()
```

### **Universal Functions (ufuncs)**

- **Use NumPy ufuncs**: They're optimized and work element-wise
- **Common ufuncs**: add, multiply, exp, log, sin, cos, etc.
- **Can specify output**: `np.add(a, b, out=result)`

**Example:**
```python
# Ufuncs are much faster than loops
arr = np.random.rand(1000000)

# Fast
result = np.sqrt(arr)

# Much slower
# result = [np.sqrt(x) for x in arr]
```

---

## Quick Reference

### **Array Creation**
- **np.array(list)**: Create array from list
- **np.zeros(shape)**: Array of zeros
- **np.ones(shape)**: Array of ones
- **np.full(shape, value)**: Array filled with value
- **np.eye(n)**: Identity matrix
- **np.arange(start, stop, step)**: Range array
- **np.linspace(start, stop, num)**: Evenly spaced array
- **np.random.rand(shape)**: Random uniform [0, 1)
- **np.random.randn(shape)**: Random normal distribution
- **np.random.randint(low, high, size)**: Random integers

### **Array Attributes**
- **array.shape**: Dimensions
- **array.size**: Total elements
- **array.ndim**: Number of dimensions
- **array.dtype**: Data type
- **array.itemsize**: Bytes per element
- **array.nbytes**: Total bytes

### **Array Manipulation**
- **array.reshape(shape)**: Change shape
- **array.ravel()**: Flatten to 1D
- **array.T**: Transpose
- **np.concatenate([a, b], axis)**: Join arrays
- **np.vstack([a, b])**: Stack vertically
- **np.hstack([a, b])**: Stack horizontally
- **np.split(array, sections)**: Split array
- **np.insert(array, index, value)**: Insert elements
- **np.delete(array, index)**: Delete elements
- **np.append(a, b)**: Append arrays

### **Indexing & Slicing**
- **array[i]**: Access element
- **array[i, j]**: Access 2D element
- **array[start:stop]**: Slice
- **array[:, col]**: Select column
- **array[row, :]**: Select row
- **array[condition]**: Boolean indexing
- **array[[indices]]**: Fancy indexing

### **Mathematical Operations**
- **array + value**: Addition
- **array * value**: Multiplication
- **array ** power**: Power
- **np.add(a, b)**: Element-wise addition
- **np.multiply(a, b)**: Element-wise multiplication
- **np.dot(a, b)**: Dot product
- **np.matmul(a, b)**: Matrix multiplication
- **np.exp(array)**: Exponential
- **np.log(array)**: Logarithm
- **np.sqrt(array)**: Square root
- **np.sin/cos/tan(array)**: Trigonometric
- **np.round/floor/ceil(array)**: Rounding

### **Statistical Operations**
- **np.sum(array)**: Sum
- **np.mean(array)**: Mean
- **np.median(array)**: Median
- **np.std(array)**: Standard deviation
- **np.var(array)**: Variance
- **np.min(array)**: Minimum
- **np.max(array)**: Maximum
- **np.argmin(array)**: Index of minimum
- **np.argmax(array)**: Index of maximum
- **np.percentile(array, q)**: Percentile
- **np.cumsum(array)**: Cumulative sum

### **Linear Algebra**
- **np.linalg.det(matrix)**: Determinant
- **np.linalg.inv(matrix)**: Inverse
- **np.linalg.eig(matrix)**: Eigenvalues
- **np.linalg.solve(A, b)**: Solve Ax=b
- **np.linalg.norm(array)**: Norm
- **np.trace(matrix)**: Trace

### **Comparison & Logic**
- **array == value**: Equality
- **array > value**: Greater than
- **array < value**: Less than
- **np.logical_and(a, b)**: AND
- **np.logical_or(a, b)**: OR
- **np.logical_not(array)**: NOT
- **np.all(array)**: All True
- **np.any(array)**: Any True
- **np.where(condition, x, y)**: Conditional selection

### **File I/O**
- **np.save('file.npy', array)**: Save binary
- **np.load('file.npy')**: Load binary
- **np.savez('file.npz', a=arr1, b=arr2)**: Save multiple
- **np.savetxt('file.txt', array)**: Save text
- **np.loadtxt('file.txt')**: Load text

---

## Important Notes for Data Engineers & Analysts

### 1. Performance Best Practices
- ✅ Always use vectorized operations instead of loops
- ✅ Use views instead of copies when possible
- ✅ Choose appropriate data types (int32 vs int64)
- ✅ Use in-place operations to save memory
- ✅ Leverage broadcasting for efficient computations
- ❌ Avoid iterating over arrays with Python loops

### 2. Memory Management
- Use smaller dtypes when possible (float32 instead of float64)
- Delete large arrays when no longer needed: `del array`
- Use memory mapping for very large files: `np.memmap()`
- Monitor memory usage with `array.nbytes`

### 3. Common Pitfalls
- **Views vs Copies**: Slicing creates views; modifications affect original
- **Array Dimensions**: Remember that `(n,)` ≠ `(n, 1)` ≠ `(1, n)`
- **Integer Division**: Use `//` for integer division, `/` gives float
- **Boolean Indexing**: Use `&` and `|` instead of `and` and `or`

### 4. Integration with Other Libraries
```python
# Convert to Pandas DataFrame
import pandas as pd
df = pd.DataFrame(numpy_array, columns=['A', 'B', 'C'])

# Convert from Pandas
numpy_array = df.to_numpy()

# Convert to PyTorch tensor
import torch
tensor = torch.from_numpy(numpy_array)

# Convert to TensorFlow tensor
import tensorflow as tf
tensor = tf.convert_to_tensor(numpy_array)
```

### 5. Debugging Tips
- Check array shape: `print(array.shape)`
- Check data type: `print(array.dtype)`
- Check for NaN: `np.isnan(array).any()`
- Check for infinity: `np.isinf(array).any()`
- Print array info: `print(np.info(array))`

---

## Conclusion

NumPy is the foundation of scientific computing in Python. Master these operations and you'll have a solid base for data science, machine learning, and numerical analysis.

### Key Takeaways
1. **Vectorization is key**: Always prefer vectorized operations over loops
2. **Understand broadcasting**: It enables efficient operations on different shapes
3. **Know views vs copies**: Avoid unintended modifications
4. **Use appropriate dtypes**: Save memory and improve performance
5. **Practice regularly**: Build intuition with real datasets

**Happy Coding! 🚀📊**