# Trendlines and Regression Analysis

# Introduction to Data Mining

# Simulation and Risk Analysis

## Excel Concepts
- Lookup Data - VLOOKUP
    - VLOOKUP(
        **value** to look up,
        **where** you want to look for it (e.g. entire table, the lookup column must be the first column), 
        **location** of the target column (counting from “ID” column),
        **return** an Approximate or Exact match (indicated as 1/TRUE, or 0/FALSE)
    )
- Lookup Data - INDEX & MATCH
    -  MATCH:
        - Input: lookup value
        - Output: the position of the value in a row/column (the output would be a number)
    - INDEX:
        - Input: the position of the return value
        - Output: return value

# Linear Optimization

# Integer and Nonlinear Optimization

# Optimiaztion Analytics

# Decision Analysis

# Python

## numpy
```python
import numpy as np

my_list = [1,2,3]
my_list
# [1, 2, 3]

np.array(my_list)
# array([1, 2, 3])

my_matrix = [[1,2,3],[4,5,6],[7,8,9]]
my_matrix
# [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

np.array(my_matrix)
# array([[1, 2, 3],
    #    [4, 5, 6],
    #    [7, 8, 9]])

'''arange'''
# > Return evenly spaced values within a given interval

np.arange(0,10)
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

np.arange(0,11,2)
# array([ 0,  2,  4,  6,  8, 10])

'''zero and ones'''
# > Generate arrays of zeros or ones. 

np.zeros(3)
# array([0., 0., 0.])

np.zeros((5,5))
# array([[0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0.]])

np.ones(3)
# array([1., 1., 1.])

np.ones((3,3))
# array([[1., 1., 1.],
#        [1., 1., 1.],
#        [1., 1., 1.]])

'''linspace'''
# > Return evenly spaced numbers over a specified interval.

np.linspace(0,10,3)
# array([ 0.,  5., 10.])

np.linspace(0,5,20)
# array([0.        , 0.26315789, 0.52631579, 0.78947368, 1.05263158,
#        1.31578947, 1.57894737, 1.84210526, 2.10526316, 2.36842105,
#        2.63157895, 2.89473684, 3.15789474, 3.42105263, 3.68421053,
#        3.94736842, 4.21052632, 4.47368421, 4.73684211, 5.        ])

# Note that .linspace() includes the stop value. To obtain an array of common fractions, increase the number of items:

np.linspace(0,5,21)
# array([0.  , 0.25, 0.5 , 0.75, 1.  , 1.25, 1.5 , 1.75, 2.  , 2.25, 2.5 ,
#        2.75, 3.  , 3.25, 3.5 , 3.75, 4.  , 4.25, 4.5 , 4.75, 5.  ])

'''eye'''
# > Creates an identity matrix

np.eye(4)
# array([[1., 0., 0., 0.],
#        [0., 1., 0., 0.],
#        [0., 0., 1., 0.],
#        [0., 0., 0., 1.]])

'''Random'''

"""
- rand -> Creates an array of the given shape and populates it with random samples from a uniform distribution over [0, 1).
- randn -> Returns a sample (or samples) from the "standard normal" distribution [σ = 1]. Unlike rand which is uniform, values closer to zero are more likely to appear. 
- randint -> Returns random integers from low (inclusive) to high (exclusive). 
"""
np.random.rand(2)
# array([0.37065108, 0.89813878])

np.random.rand(5,5)
# array([[0.03932992, 0.80719137, 0.50145497, 0.68816102, 0.1216304 ],
#        [0.44966851, 0.92572848, 0.70802042, 0.10461719, 0.53768331],
#        [0.12201904, 0.5940684 , 0.89979774, 0.3424078 , 0.77421593],
#        [0.53191409, 0.0112285 , 0.3989947 , 0.8946967 , 0.2497392 ],
#        [0.5814085 , 0.37563686, 0.15266028, 0.42948309, 0.26434141]])

np.random.randn(2)
# array([-0.36633217, -1.40298731])

np.random.randn(5,5)
# array([[-0.45241033,  1.07491082,  1.95698188,  0.40660223, -1.50445807],
#        [ 0.31434506, -2.16912609, -0.51237235,  0.78663583, -0.61824678],
#        [-0.17569928, -2.39139828,  0.30905559,  0.1616695 ,  0.33783857],
#        [-0.2206597 , -0.05768918,  0.74882883, -1.01241629, -1.81729966],
#        [-0.74891671,  0.88934796,  1.32275912, -0.71605188,  0.0450718 ]])

np.random.randint(1,100)
# 61

np.random.randint(1,100,10)
# array([39, 50, 72, 18, 27, 59, 15, 97, 11, 14])

'''seed'''
np.random.seed(42)
np.random.rand(4)
array([0.37454012, 0.95071431, 0.73199394, 0.59865848])

'''Reshape'''

# suppose arr was one row with 25 numbers
arr.reshape(5,5)
# array([[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14],
#        [15, 16, 17, 18, 19],
#        [20, 21, 22, 23, 24]])

'''max, min, argmax, argmin'''
# > hese are useful methods for finding max or min values. Or to find their index locations using argmin or argmax

ranarr # suppose it contains: array([38, 18, 22, 10, 10, 23, 35, 39, 23,  2])

ranarr.max()
# 39

ranarr.argmax()
# 7

ranarr.min()
# 2

ranarr.argmin()
# 9

'''Shape'''
# > Shape is an attribute that arrays have (not a method): 

# Vector
arr.shape
(25,)

# Notice the two sets of brackets
arr.reshape(1,25)
# array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
#         16, 17, 18, 19, 20, 21, 22, 23, 24]])

arr.reshape(1,25).shape
# (1, 25)

arr.reshape(25,1)
# array([[ 0],
#        [ 1],
#        [ 2],
#        [ 3],
#        [ 4],
#        [ 5],
#        [ 6],
#        [ 7],
#        [ 8],
#        [ 9],
#        [10],
#        [11],
#        [12],
#        [13],
#        [14],
#        [15],
#        [16],
#        [17],
#        [18],
#        [19],
#        [20],
#        [21],
#        [22],
#        [23],
#        [24]]) 

```
- [arange document](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.arange.html)
- [zeros](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.zeros.html)
- [linspace](https://numpy.org/devdocs/reference/generated/numpy.linspace.html)
- [eye](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.eye.html)
- [rand](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.rand.html)
- [randn](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.randn.html)
- [randint](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.randint.html)
- [seed](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.seed.html)
- [reshape](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.reshape.html)
- [shape](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.shape.html)
- [dtype](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.ndarray.dtype.html)

## numpy - indexing and selection

```python
import numpy as np

#Creating sample array
arr = np.arange(0,11)
#Show
arr
# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

'''bracket indexing and selection'''

#Get a value at an index
arr[8]
# 8

#Get values in a range
arr[0:5]
# array([0, 1, 2, 3, 4])

# NumPy arrays differ from normal Python lists because of their ability to broadcast. With lists, you can only reassign parts of a list with new parts of the same size and shape. That is, if you wanted to replace the first 5 elements in a list with a new value, you would have to pass in a new 5 element list. With NumPy arrays, you can broadcast a single value across a larger set of values:

#Setting a value with index range (Broadcasting)
arr[0:5]=100
#Show
arr
# array([100, 100, 100, 100, 100,   5,   6,   7,   8,   9,  10])

# Reset array, we'll see why I had to reset in  a moment
arr = np.arange(0,11)
#Show
arr
# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

#Important notes on Slices
slice_of_arr = arr[0:6]
#Show slice
slice_of_arr
# array([0, 1, 2, 3, 4, 5])

#Change Slice
slice_of_arr[:]=99
#Show Slice again
slice_of_arr
# array([99, 99, 99, 99, 99, 99])

# Now note the changes also occur in our original array!
arr
# array([99, 99, 99, 99, 99, 99,  6,  7,  8,  9, 10])

# Data is not copied, it's a view of the original array! This avoids memory problems!
# To get a copy, need to be explicit
arr_copy = arr.copy()
arr_copy
# array([99, 99, 99, 99, 99, 99,  6,  7,  8,  9, 10])

'''indexing 2d array'''
arr_2d = np.array(([5,10,15],[20,25,30],[35,40,45]))

#Show
arr_2d
# array([[ 5, 10, 15],
#        [20, 25, 30],
#        [35, 40, 45]])

#Indexing row
arr_2d[1]
# array([20, 25, 30])

# Format is arr_2d[row][col] or arr_2d[row,col]
# Getting individual element value
arr_2d[1][0]
# 20

# Getting individual element value
arr_2d[1,0]
# 20

# 2D array slicing
#Shape (2,2) from top right corner
arr_2d[:2,1:]
# array([[10, 15],
#        [25, 30]])

#Shape bottom row
arr_2d[2]
# array([35, 40, 45])

#Shape bottom row
arr_2d[2,:]
# array([35, 40, 45])

'''Condictional Selection'''
arr = np.arange(1,11)
arr
# array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

arr > 4
# array([False, False, False, False,  True,  True,  True,  True,  True,
#         True])

bool_arr = arr > 4
bool_arr
# array([False, False, False, False,  True,  True,  True,  True,  True,
#         True])

arr[bool_arr]
# array([ 5,  6,  7,  8,  9, 10])

arr[arr>2]
# array([ 3,  4,  5,  6,  7,  8,  9, 10])

x = 2
arr[arr>x]
# array([ 3,  4,  5,  6,  7,  8,  9, 10])
```

## numpy - operation

```python
import numpy as np
'''arthmetic'''
arr = np.arange(0,10)
arr
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

arr + arr
# array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])

arr * arr
# array([ 0,  1,  4,  9, 16, 25, 36, 49, 64, 81])

arr - arr
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

arr + 5
# array([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14])

# This will raise a Warning on division by zero, but not an error!
# It just fills the spot with nan
arr/arr
# array([nan,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])

# Also a warning (but not an error) relating to infinity
1/arr
# array([       inf, 1.        , 0.5       , 0.33333333, 0.25      ,
#        0.2       , 0.16666667, 0.14285714, 0.125     , 0.11111111])

arr**3
# array([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729], dtype=int32)


'''universal array function'''
# Taking Square Roots
np.sqrt(arr)
# array([0.        , 1.        , 1.41421356, 1.73205081, 2.        ,
#        2.23606798, 2.44948974, 2.64575131, 2.82842712, 3.        ])

# Calculating exponential (e^)
np.exp(arr)
# array([1.00000000e+00, 2.71828183e+00, 7.38905610e+00, 2.00855369e+01,
#        5.45981500e+01, 1.48413159e+02, 4.03428793e+02, 1.09663316e+03,
#        2.98095799e+03, 8.10308393e+03])

# Trigonometric Functions like sine
np.sin(arr)
# array([ 0.        ,  0.84147098,  0.90929743,  0.14112001, -0.7568025 ,
#        -0.95892427, -0.2794155 ,  0.6569866 ,  0.98935825,  0.41211849])

# Taking the Natural Logarithm
np.log(arr)
# array([      -inf, 0.        , 0.69314718, 1.09861229, 1.38629436,
#        1.60943791, 1.79175947, 1.94591015, 2.07944154, 2.19722458])


'''summary statistics on arrays'''
# > NumPy also offers common summary statistics like sum, mean and max. You would call these as methods on an array.
arr = np.arange(0,10)
arr
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

arr.sum()
# 45

arr.mean()
# 4.5

arr.max()
# 9

# Other summary statistics include:

arr.min() # returns 0                   minimum
arr.var() # returns 8.25                variance
arr.std() # returns 2.8722813232690143  standard deviation

'''axis logic'''
# > When working with 2-dimensional arrays (matrices) we have to consider rows and columns. This becomes very important when we get to the section on pandas. In array terms, axis 0 (zero) is the vertical axis (rows), and axis 1 is the horizonal axis (columns). These values (0,1) correspond to the order in which arr.shape values are returned.

Let's see how this affects our summary statistic calculations from above.
arr_2d = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
arr_2d
# array([[ 1,  2,  3,  4],
#        [ 5,  6,  7,  8],
#        [ 9, 10, 11, 12]])

arr_2d.sum(axis=0)
# array([15, 18, 21, 24])

arr_2d.sum(axis=1)
# array([10, 26, 42])
```
- [Universal Array Function](https://numpy.org/doc/stable/reference/ufuncs.html)


## pandas - series (part one & part two)
```python
import pandas as pd
import numpy as np # create some data for educational purpose

help(pd.Series) # give you documentation when using help()

'''index and data list'''
# we can create a series from python list
myindex = ['USA','Canada','Mexico']
mydata = [1776,1867,1821]

myser = pd.Series(data=mydata)
myser
# 0    1776
# 1    1867
# 2    1821
# dtype: int64

pd.Series(data=mydata,index=myindex)
# USA       1776
# Canada    1867
# Mexico    1821
# dtype: int64

ran_data = np.random.randint(0,100,4)
ran_data
# array([39, 35, 37, 23])

names = ['Andrew','Bobo','Claire','David']
ages = pd.Series(ran_data,names)
ages
# Andrew    39
# Bobo      35
# Claire    37
# David     23
# dtype: int32

'''from a dictionary'''
ages = {'Sammy':5,'Frank':10,'Spike':7}
pd.Series(ages)
# Sammy     5
# Frank    10
# Spike     7
# dtype: int64

'''key idea of a series'''

# Imaginary Sales Data for 1st and 2nd Quarters for Global Company
q1 = {'Japan': 80, 'China': 450, 'India': 200, 'USA': 250}
q2 = {'Brazil': 100,'China': 500, 'India': 210,'USA': 260}

# Convert into Pandas Series
sales_Q1 = pd.Series(q1)
sales_Q2 = pd.Series(q2)
sales_Q1
# Japan     80
# China    450
# India    200
# USA      250
# dtype: int64

# Call values based on Named Index
sales_Q1['Japan']
# 80

# Integer Based Location information also retained!
sales_Q1[0]
# 80

'''operation'''

# Grab just the index keys
sales_Q1.keys()
# Index(['Japan', 'China', 'India', 'USA'], dtype='object')

# Can Perform Operations Broadcasted across entire Series
sales_Q1 * 2
# Japan    160
# China    900
# India    400
# USA      500
# dtype: int64

sales_Q2 / 100
# Brazil    1.0
# China     5.0
# India     2.1
# USA       2.6
# dtype: float64

'''Between Series'''

# Notice how Pandas informs you of mismatch with NaN
sales_Q1 + sales_Q2
# Brazil      NaN
# China     950.0
# India     410.0
# Japan       NaN
# USA       510.0
# dtype: float64

# You can fill these with any value you want
sales_Q1.add(sales_Q2,fill_value=0)
# Brazil    100.0
# China     950.0
# India     410.0
# Japan      80.0
# USA       510.0
# dtype: float64
```
