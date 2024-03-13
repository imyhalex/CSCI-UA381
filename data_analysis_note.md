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

- Goal Seek
    - **Input:** the desired result from a formula
    - **Output:** the formula input value that produces this formula result

- Solver
    - **Input:**
        - Objective (maximize, minimize, or to a certain value)
        - Constraints
    - **Output:** 
        - Optimal values of decision variables that achieves the objective

## Sensitivity Analysis
> **Sensitivity Analysis** is used to understand the effect of a set of independent variables on some dependent variable under certain specific conditions.  

- **input:** a given set of assumptions
- **output:** possible outcomes of the objective 

### Sensitivity Analysis Tools
1. Data Table
2. Tornado Chart
    - Building from scratch
    - Senslt Add-in
3. Solver Sensitivity Report

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

## pandas - creating a dataframe
```python
import numpy as np # create some data for educational purpose
import pandas as pd

# Make sure the seed is in the same cell as the random call
# https://stackoverflow.com/questions/21494489/what-does-numpy-random-seed0-do
np.random.seed(101)
mydata = np.random.randint(0,101,(4,3))
mydata
# array([[95, 11, 81],
#        [70, 63, 87],
#        [75,  9, 77],
#        [40,  4, 63]])

myindex = ['CA','NY','AZ','TX']
mycolumns = ['Jan','Feb','Mar']
df = pd.DataFrame(data=mydata)
df
# 	0	1	2
# 0	95	11	81
# 1	70	63	87
# 2	75	9	77
# 3	40	4	63

df = pd.DataFrame(data=mydata,index=myindex)
df
# 	0	1	2
# CA	95	11	81
# NY	70	63	87
# AZ	75	9	77
# TX	40	4	63

df = pd.DataFrame(data=mydata,index=myindex,columns=mycolumns)
df 
# 	Jan	Feb	Mar
# CA	95	11	81
# NY	70	63	87
# AZ	75	9	77
# TX	40	4	63

df.info()
# <class 'pandas.core.frame.DataFrame'>
# Index: 4 entries, CA to TX
# Data columns (total 3 columns):
# Jan    4 non-null int32
# Feb    4 non-null int32
# Mar    4 non-null int32
# dtypes: int32(3)
# memory usage: 80.0+ bytes


'''Readning a .csv file for a dataframe'''
df = pd.read_csv('tips.csv')
df
# give you the data from 'tip.csv'

'''obtain basic information about dataframe'''
df.columns
# Index(['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size',
#        'price_per_person', 'Payer Name', 'CC Number', 'Payment ID'],
#       dtype='object')

df.index
# RangeIndex(start=0, stop=244, step=1)

df.head(3)
# 	total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID
# 0	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410	Sun2959
# 1	10.34	1.66	Male	No	Sun	Dinner	3	3.45	Douglas Tucker	    4478071379779230	Sun4608
# 2	21.01	3.50	Male	No	Sun	Dinner	3	7.00	Travis Walters	    6011812112971322	Sun4458

df.tail(3)
# 	total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID
# 241	22.67	2.00	Male	Yes	Sat	Dinner	2	11.34	Keith Wong	6011891618747196	Sat3880
# 242	17.82	1.75	Male	No	Sat	Dinner	2	8.91	Dennis Dixon	4375220550950	Sat17
# 243	18.78	3.00	Female	No	Thur	Dinner	2	9.39	Michelle Hardin	3511451626698139	Thur672

len(df)
# 244

df.describe()
# total_bill	tip	size	price_per_person	        CC Number
# count	    244.000000	244.000000	244.000000	244.000000	2.440000e+02
# mean	    19.785943	2.998279	2.569672	7.888197	2.563496e+15
# std	    8.902412	1.383638	0.951100	2.914234	2.369340e+15
# min	    3.070000	1.000000	1.000000	2.880000	6.040679e+10
# 25%	    13.347500	2.000000	2.000000	5.800000	3.040731e+13
# 50%	    17.795000	2.900000	2.000000	7.255000	3.525318e+15
# 75%	    24.127500	3.562500	3.000000	9.390000	4.553675e+15
# max	    50.810000	10.000000	6.000000	20.270000	6.596454e+15

df.describe().transpose()
# 	count	mean	std	min	25%	50%	75%	max
# total_bill	244.0	1.978594e+01	8.902412e+00	3.070000e+00	1.334750e+01	1.779500e+01	2.412750e+01	5.081000e+01
# tip	244.0	2.998279e+00	1.383638e+00	1.000000e+00	2.000000e+00	2.900000e+00	3.562500e+00	1.000000e+01
# size	244.0	2.569672e+00	9.510998e-01	1.000000e+00	2.000000e+00	2.000000e+00	3.000000e+00	6.000000e+00
# price_per_person	244.0	7.888197e+00	2.914234e+00	2.880000e+00	5.800000e+00	7.255000e+00	9.390000e+00	2.027000e+01
# CC Number	244.0	2.563496e+15	2.369340e+15	6.040679e+10	3.040731e+13	3.525318e+15	4.553675e+15	6.596454e+15

'''selection and indexing'''

df.head()
# total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID
# 0	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410	Sun2959
# 1	10.34	1.66	Male	No	Sun	Dinner	3	3.45	Douglas Tucker	4478071379779230	Sun4608
# 2	21.01	3.50	Male	No	Sun	Dinner	3	7.00	Travis Walters	6011812112971322	Sun4458
# 3	23.68	3.31	Male	No	Sun	Dinner	2	11.84	Nathaniel Harris	4676137647685994	Sun5260
# 4	24.59	3.61	Female	No	Sun	Dinner	4	6.15	Tonya Carter	4832732618637221	Sun2251

# grab a single column
df['total_bill']
# 0      16.99
# 1      10.34
# 2      21.01
# 3      23.68
# 4      24.59
# 5      25.29
# 6       8.77
# 7      26.88
# 8      15.04
# 9      14.78
# 10     10.27
# 11     35.26
# 12     15.42
# 13     18.43
# 14     14.83
# 15     21.58
# 16     10.33
# 17     16.29
# 18     16.97
# 19     20.65
# 20     17.92
# 21     20.29
# 22     15.77
# 23     39.42
# 24     19.82
# 25     17.81
# 26     13.37
# 27     12.69
# 28     21.70
# 29     19.65
#        ...  
# 214    28.17
# 215    12.90
# 216    28.15
# 217    11.59
# 218     7.74
# 219    30.14
# 220    12.16
# 221    13.42
# 222     8.58
# 223    15.98
# 224    13.42
# 225    16.27
# 226    10.09
# 227    20.45
# 228    13.28
# 229    22.12
# 230    24.01
# 231    15.69
# 232    11.61
# 233    10.77
# 234    15.53
# 235    10.07
# 236    12.60
# 237    32.83
# 238    35.83
# 239    29.03
# 240    27.18
# 241    22.67
# 242    17.82
# 243    18.78
# Name: total_bill, Length: 244, dtype: float64

type(df['total_bill'])
# pandas.core.series.Series

# grab multiple columns
# Note how its a python list of column names! Thus the double brackets.
df[['total_bill','tip']]
	total_bill	tip
# 0	16.99	1.01
# 1	10.34	1.66
# 2	21.01	3.50
# 3	23.68	3.31
# 4	24.59	3.61
# 5	25.29	4.71
# 6	8.77	2.00
# 7	26.88	3.12
# 8	15.04	1.96
# 9	14.78	3.23
# ...	...	...
# 231	15.69	3.00
# 232	11.61	3.39
# 233	10.77	1.47
# 234	15.53	3.00
# 235	10.07	1.25
# 236	12.60	1.00
# 237	32.83	1.17
# 238	35.83	4.67
# 239	29.03	5.92
# 240	27.18	2.00
# 241	22.67	2.00
# 242	17.82	1.75
# 243	18.78	3.00

# create new column
df['tip_percentage'] = 100* df['tip'] / df['total_bill']
df.head()
# 	total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID	tip_percentage
# 0	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410	Sun2959	5.944673
# 1	10.34	1.66	Male	No	Sun	Dinner	3	3.45	Douglas Tucker	4478071379779230	Sun4608	16.054159
# 2	21.01	3.50	Male	No	Sun	Dinner	3	7.00	Travis Walters	6011812112971322	Sun4458	16.658734
# 3	23.68	3.31	Male	No	Sun	Dinner	2	11.84	Nathaniel Harris	4676137647685994	Sun5260	13.978041
# 4	24.59	3.61	Female	No	Sun	Dinner	4	6.15	Tonya Carter	4832732618637221	Sun2251	14.680765

df['price_per_person'] = df['total_bill'] / df['size']
df.head()
# total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID	tip_percentage
# 0	16.99	1.01	Female	No	Sun	Dinner	2	8.495000	Christy Cunningham	3560325168603410	Sun2959	5.944673
# 1	10.34	1.66	Male	No	Sun	Dinner	3	3.446667	Douglas Tucker	4478071379779230	Sun4608	16.054159
# 2	21.01	3.50	Male	No	Sun	Dinner	3	7.003333	Travis Walters	6011812112971322	Sun4458	16.658734
# 3	23.68	3.31	Male	No	Sun	Dinner	2	11.840000	Nathaniel Harris	4676137647685994	Sun5260	13.978041
# 4	24.59	3.61	Female	No	Sun	Dinner	4	6.147500	Tonya Carter	4832732618637221	Sun2251	14.680765

'''adjusting existing columns'''
# Because pandas is based on numpy, we get awesome capabilities with numpy's universal functions!
df['price_per_person'] = np.round(df['price_per_person'],2)
df.head()
# total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID	tip_percentage
# 0	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410	Sun2959	5.944673
# 1	10.34	1.66	Male	No	Sun	Dinner	3	3.45	Douglas Tucker	4478071379779230	Sun4608	16.054159
# 2	21.01	3.50	Male	No	Sun	Dinner	3	7.00	Travis Walters	6011812112971322	Sun4458	16.658734
# 3	23.68	3.31	Male	No	Sun	Dinner	2	11.84	Nathaniel Harris	4676137647685994	Sun5260	13.978041
# 4	24.59	3.61	Female	No	Sun	Dinner	4	6.15	Tonya Carter	4832732618637221	Sun2251	14.680765

'''remove column'''
df = df.drop("tip_percentage",axis=1)
df.head()
# total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID
# 0	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410	Sun2959
# 1	10.34	1.66	Male	No	Sun	Dinner	3	3.45	Douglas Tucker	4478071379779230	Sun4608
# 2	21.01	3.50	Male	No	Sun	Dinner	3	7.00	Travis Walters	6011812112971322	Sun4458
# 3	23.68	3.31	Male	No	Sun	Dinner	2	11.84	Nathaniel Harris	4676137647685994	Sun5260
# 4	24.59	3.61	Female	No	Sun	Dinner	4	6.15	Tonya Carter	4832732618637221	Sun2251

'''index basic'''
df.index
# RangeIndex(start=0, stop=244, step=1)

df.set_index('Payment ID')
#         total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number
# Payment ID										
# Sun2959	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410
# Sun4608	10.34	1.66	Male	No	Sun	Dinner	3	3.45	Douglas Tucker	4478071379779230
# Sun4458	21.01	3.50	Male	No	Sun	Dinner	3	7.00	Travis Walters	6011812112971322
# Sun5260	23.68	3.31	Male	No	Sun	Dinner	2	11.84	Nathaniel Harris	4676137647685994
# Sun2251	24.59	3.61	Female	No	Sun	Dinner	4	6.15	Tonya Carter	4832732618637221
# Sun9679	25.29	4.71	Male	No	Sun	Dinner	4	6.32	Erik Smith	213140353657882

df = df.reset_index()
df.head()
#   Payment ID	total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number
# 0	Sun2959	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410
# 1	Sun4608	10.34	1.66	Male	No	Sun	Dinner	3	3.45	Douglas Tucker	4478071379779230
# 2	Sun4458	21.01	3.50	Male	No	Sun	Dinner	3	7.00	Travis Walters	6011812112971322
# 3	Sun5260	23.68	3.31	Male	No	Sun	Dinner	2	11.84	Nathaniel Harris	4676137647685994
# 4	Sun2251	24.59	3.61	Female	No	Sun	Dinner	4	6.15	Tonya Carter	4832732618637221

'''rows'''
df = df.set_index('Payment ID')
df.head()
#           total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number
# Payment ID										
# Sun2959	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410
# Sun4608	10.34	1.66	Male	No	Sun	Dinner	3	3.45	Douglas Tucker	4478071379779230
# Sun4458	21.01	3.50	Male	No	Sun	Dinner	3	7.00	Travis Walters	6011812112971322
# Sun5260	23.68	3.31	Male	No	Sun	Dinner	2	11.84	Nathaniel Harris	4676137647685994
# Sun2251	24.59	3.61	Female	No	Sun	Dinner	4	6.15	Tonya Carter	4832732618637221

# Integer Based
df.iloc[0] # index label index
# total_bill                       16.99
# tip                               1.01
# sex                             Female
# smoker                              No
# day                                Sun
# time                            Dinner
# size                                 2
# price_per_person                  8.49
# Payer Name          Christy Cunningham
# CC Number             3560325168603410
# Name: Sun2959, dtype: object

# Name Based
df.loc['Sun2959'] # label index
# total_bill                       16.99
# tip                               1.01
# sex                             Female
# smoker                              No
# day                                Sun
# time                            Dinner
# size                                 2
# price_per_person                  8.49
# Payer Name          Christy Cunningham
# CC Number             3560325168603410
# Name: Sun2959, dtype: object

df.iloc[0:4]
# 	total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number
# Payment ID										
# Sun2959	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410
# Sun4608	10.34	1.66	Male	No	Sun	Dinner	3	3.45	Douglas Tucker	4478071379779230
# Sun4458	21.01	3.50	Male	No	Sun	Dinner	3	7.00	Travis Walters	6011812112971322
# Sun5260	23.68	3.31	Male	No	Sun	Dinner	2	11.84	Nathaniel Harris	4676137647685994

df.loc[['Sun2959','Sun5260']]
# 	        total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number
# Payment ID										
# Sun2959	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410
# Sun5260	23.68	3.31	Male	No	Sun	Dinner	2	11.84	Nathaniel Harris	4676137647685994

'''remove row'''
df.drop('Sun2959',axis=0).head()
# 	total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number
# Payment ID										
# Sun4608	10.34	1.66	Male	No	Sun	Dinner	3	3.45	Douglas Tucker	4478071379779230
# Sun4458	21.01	3.50	Male	No	Sun	Dinner	3	7.00	Travis Walters	6011812112971322
# Sun5260	23.68	3.31	Male	No	Sun	Dinner	2	11.84	Nathaniel Harris	4676137647685994
# Sun2251	24.59	3.61	Female	No	Sun	Dinner	4	6.15	Tonya Carter	4832732618637221
# Sun9679	25.29	4.71	Male	No	Sun	Dinner	4	6.32	Erik Smith	213140353657882

# Error if you have a named index!
# df.drop(0,axis=0).head()

'''insert new row'''
one_row = df.iloc[0]
one_row
# total_bill                       16.99
# tip                               1.01
# sex                             Female
# smoker                              No
# day                                Sun
# time                            Dinner
# size                                 2
# price_per_person                  8.49
# Payer Name          Christy Cunningham
# CC Number             3560325168603410
# Name: Sun2959, dtype: object


df.tail()
# 	total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number
# Payment ID										
# Sat2657	29.03	5.92	Male	No	Sat	Dinner	3	9.68	Michael Avila	5296068606052842
# Sat1766	27.18	2.00	Female	Yes	Sat	Dinner	2	13.59	Monica Sanders	3506806155565404
# Sat3880	22.67	2.00	Male	Yes	Sat	Dinner	2	11.34	Keith Wong	6011891618747196
# Sat17	17.82	1.75	Male	No	Sat	Dinner	2	8.91	Dennis Dixon	4375220550950
# Thur672	18.78	3.00	Female	No	Thur	Dinner	2	9.39	Michelle Hardin	3511451626698139

df.append(one_row).tail()
# 	total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number
# Payment ID										
# Sat1766	27.18	2.00	Female	Yes	Sat	Dinner	2	13.59	Monica Sanders	3506806155565404
# Sat3880	22.67	2.00	Male	Yes	Sat	Dinner	2	11.34	Keith Wong	6011891618747196
# Sat17	17.82	1.75	Male	No	Sat	Dinner	2	8.91	Dennis Dixon	4375220550950
# Thur672	18.78	3.00	Female	No	Thur	Dinner	2	9.39	Michelle Hardin	3511451626698139
# Sun2959	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410
```

## pandas - condictional filtering
```python
import pandas as pd

df = pd.read_csv('tips.csv')
df.head()
# 	total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID
# 0	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410	Sun2959
# 1	10.34	1.66	Male	No	Sun	Dinner	3	3.45	Douglas Tucker	4478071379779230	Sun4608
# 2	21.01	3.50	Male	No	Sun	Dinner	3	7.00	Travis Walters	6011812112971322	Sun4458
# 3	23.68	3.31	Male	No	Sun	Dinner	2	11.84	Nathaniel Harris	4676137647685994	Sun5260
# 4	24.59	3.61	Female	No	Sun	Dinner	4	6.15	Tonya Carter	4832732618637221	Sun2251

'''conditions'''
bool_series = df['total_bill'] > 30
df[bool_series]
# 	total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID
# 11	35.26	5.00	Female	No	Sun	Dinner	4	8.82	Diane Macias	4577817359320969	Sun6686
# 23	39.42	7.58	Male	No	Sat	Dinner	4	9.86	Lance Peterson	3542584061609808	Sat239
# 39	31.27	5.00	Male	No	Sat	Dinner	3	10.42	Mr. Brandon Berry	6011525851069856	Sat6373
# 44	30.40	5.60	Male	No	Sun	Dinner	4	7.60	Todd Cooper	503846761263	Sun2274
# 47	32.40	6.00	Male	No	Sun	Dinner	4	8.10	James Barnes	3552002592874186	Sun9677
# ...

# or we can do
df[df['total_bill']>30]
# 	total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID
# 11	35.26	5.00	Female	No	Sun	Dinner	4	8.82	Diane Macias	4577817359320969	Sun6686
# 23	39.42	7.58	Male	No	Sat	Dinner	4	9.86	Lance Peterson	3542584061609808	Sat239
# 39	31.27	5.00	Male	No	Sat	Dinner	3	10.42	Mr. Brandon Berry	6011525851069856	Sat6373
# 44	30.40	5.60	Male	No	Sun	Dinner	4	7.60	Todd Cooper	503846761263	Sun2274
# 47	32.40	6.00	Male	No	Sun	Dinner	4	8.10	James Barnes	3552002592874186	Sun9677
# ...

# more condictional example
df[df['sex'] == 'Male']

'''multiple conditions'''
"""
| -> or
& -> and
~ -> not
"""

df[(df['total_bill'] > 30) & (df['sex']=='Male')]
# 	total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID
# 23	39.42	7.58	Male	No	Sat	Dinner	4	9.86	Lance Peterson	3542584061609808	Sat239
# 39	31.27	5.00	Male	No	Sat	Dinner	3	10.42	Mr. Brandon Berry	6011525851069856	Sat6373
# 44	30.40	5.60	Male	No	Sun	Dinner	4	7.60	Todd Cooper	503846761263	Sun2274
# 47	32.40	6.00	Male	No	Sun	Dinner	4	8.10	James Barnes	3552002592874186	Sun9677
# 56	38.01	3.00	Male	Yes	Sat	Dinner	4	9.50	James Christensen DDS	349793629453226	Sat8903
# 59	48.27	6.73	Male	No	Sat	Dinner	4	12.07	Brian Ortiz	6596453823950595	Sat8139
# ...

df[(df['total_bill'] > 30) & ~(df['sex']=='Male')]
# total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID
# 11	35.26	5.00	Female	No	Sun	Dinner	4	8.82	Diane Macias	4577817359320969	Sun6686
# 52	34.81	5.20	Female	No	Sun	Dinner	4	8.70	Emily Daniel	4291280793094374	Sun6165
# 85	34.83	5.17	Female	No	Thur	Lunch	4	8.71	Shawna Cook	6011787464177340	Thur7972
# 102	44.30	2.50	Female	Yes	Sat	Dinner	3	14.77	Heather Cohen	379771118886604	Sat6240
# 197	43.11	5.00	Female	Yes	Thur	Lunch	4	10.78	Brooke Soto	5544902205760175	Thur9313
# 219	30.14	3.09	Female	Yes	Sat	Dinner	4	7.54	Shelby House	502097403252	Sat8863
# 238	35.83	4.67	Female	No	Sat	Dinner	3	11.94	Kimberly Crane	676184013727	Sat9777

# more examples
df[(df['total_bill'] > 30) & (df['sex']!='Male')]
df[(df['day'] =='Sun') | (df['day']=='Sat')]


'''conditional operator isin()'''
options = ['Sat','Sun']
df['day'].isin(options)
# 0       True
# 1       True
# 2       True
# 3       True
# 4       True
# 5       True
# 6       True
# 7       True
# 8       True
# 9       True
# 10      True
# 11      True
# 12      True
# 13      True
# ...
# Name: day, Length: 244, dtype: bool

df[df['day'].isin(['Sat','Sun'])]
# total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID
# 0	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410	Sun2959
# 1	10.34	1.66	Male	No	Sun	Dinner	3	3.45	Douglas Tucker	4478071379779230	Sun4608
# 2	21.01	3.50	Male	No	Sun	Dinner	3	7.00	Travis Walters	6011812112971322	Sun4458
# 3	23.68	3.31	Male	No	Sun	Dinner	2	11.84	Nathaniel Harris	4676137647685994	Sun5260
# 4	24.59	3.61	Female	No	Sun	Dinner	4	6.15	Tonya Carter	4832732618637221	Sun2251
# ...
```

## pandas - useful methods - apply on single column
```python
import pandas as pd
import numpy as np

df = pd.read_csv('tips.csv')
df.head()
# 	total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID
# 0	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410	Sun2959
# 1	10.34	1.66	Male	No	Sun	Dinner	3	3.45	Douglas Tucker	4478071379779230	Sun4608
# 2	21.01	3.50	Male	No	Sun	Dinner	3	7.00	Travis Walters	6011812112971322	Sun4458
# 3	23.68	3.31	Male	No	Sun	Dinner	2	11.84	Nathaniel Harris	4676137647685994	Sun5260
# 4	24.59	3.61	Female	No	Sun	Dinner	4	6.15	Tonya Carter	4832732618637221	Sun2251

def last_four(num):
    return str(num)[-4:]

df['CC Number'][0]
# 3560325168603410

last_four(3560325168603410)
# '3410'

df['last_four'] = df['CC Number'].apply(last_four)
df.head()
# total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID	last_four
# 0	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410	Sun2959	3410
# 1	10.34	1.66	Male	No	Sun	Dinner	3	3.45	Douglas Tucker	4478071379779230	Sun4608	9230
# 2	21.01	3.50	Male	No	Sun	Dinner	3	7.00	Travis Walters	6011812112971322	Sun4458	1322
# 3	23.68	3.31	Male	No	Sun	Dinner	2	11.84	Nathaniel Harris	4676137647685994	Sun5260	5994
# 4	24.59	3.61	Female	No	Sun	Dinner	4	6.15	Tonya Carter	4832732618637221	Sun2251	7221

df['total_bill'].mean()
# 19.78594262295082

def yelp(price):
    if price < 10:
        return '$'
    elif price >= 10 and price < 30:
        return '$$'
    else:
        return '$$$'
df['Expensive'] = df['total_bill'].apply(yelp)
df.head()
# total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID	last_four	Expensive
# 0	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410	Sun2959	3410	$$
# 1	10.34	1.66	Male	No	Sun	Dinner	3	3.45	Douglas Tucker	4478071379779230	Sun4608	9230	$$
# 2	21.01	3.50	Male	No	Sun	Dinner	3	7.00	Travis Walters	6011812112971322	Sun4458	1322	$$
# 3	23.68	3.31	Male	No	Sun	Dinner	2	11.84	Nathaniel Harris	4676137647685994	Sun5260	5994	$$
# 4	24.59	3.61	Female	No	Sun	Dinner	4	6.15	Tonya Carter	4832732618637221	Sun2251	7221	$$

'''apply with lambda'''

df['total_bill'].apply(lambda bill:bill*0.18)

'''apply that uses multiple columns'''
def quality(total_bill,tip):
    if tip/total_bill  > 0.25:
        return "Generous"
    else:
        return "Other"
df['Tip Quality'] = df[['total_bill','tip']].apply(lambda df: quality(df['total_bill'],df['tip']),axis=1) # slower
df.head()
# total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID	last_four	Expensive	Tip Quality
# 0	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410	Sun2959	3410	$$	Other
# 1	10.34	1.66	Male	No	Sun	Dinner	3	3.45	Douglas Tucker	4478071379779230	Sun4608	9230	$$	Other
# 2	21.01	3.50	Male	No	Sun	Dinner	3	7.00	Travis Walters	6011812112971322	Sun4458	1322	$$	Other
# 3	23.68	3.31	Male	No	Sun	Dinner	2	11.84	Nathaniel Harris	4676137647685994	Sun5260	5994	$$	Other
# 4	24.59	3.61	Female	No	Sun	Dinner	4	6.15	Tonya Carter	4832732618637221	Sun2251	7221	$$	Other

df['Tip Quality'] = np.vectorize(quality)(df['total_bill'], df['tip']) # way more faster
df.head()
# total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID	last_four	Expensive	Tip Quality
# 0	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410	Sun2959	3410	$$	Other
# 1	10.34	1.66	Male	No	Sun	Dinner	3	3.45	Douglas Tucker	4478071379779230	Sun4608	9230	$$	Other
# 2	21.01	3.50	Male	No	Sun	Dinner	3	7.00	Travis Walters	6011812112971322	Sun4458	1322	$$	Other
# 3	23.68	3.31	Male	No	Sun	Dinner	2	11.84	Nathaniel Harris	4676137647685994	Sun5260	5994	$$	Other
# 4	24.59	3.61	Female	No	Sun	Dinner	4	6.15	Tonya Carter	4832732618637221	Sun2251	7221	$$	Other

df.describe()
# 	total_bill	tip	size	price_per_person	CC Number
# count	244.000000	244.000000	244.000000	244.000000	2.440000e+02
# mean	19.785943	2.998279	2.569672	7.888197	2.563496e+15
# std	8.902412	1.383638	0.951100	2.914234	2.369340e+15
# min	3.070000	1.000000	1.000000	2.880000	6.040679e+10
# 25%	13.347500	2.000000	2.000000	5.800000	3.040731e+13
# 50%	17.795000	2.900000	2.000000	7.255000	3.525318e+15
# 75%	24.127500	3.562500	3.000000	9.390000	4.553675e+15
# max	50.810000	10.000000	6.000000	20.270000	6.596454e+15

df.describe().transpose()
# count	mean	std	min	25%	50%	75%	max
# total_bill	244.0	1.978594e+01	8.902412e+00	3.070000e+00	1.334750e+01	1.779500e+01	2.412750e+01	5.081000e+01
# tip	244.0	2.998279e+00	1.383638e+00	1.000000e+00	2.000000e+00	2.900000e+00	3.562500e+00	1.000000e+01
# size	244.0	2.569672e+00	9.510998e-01	1.000000e+00	2.000000e+00	2.000000e+00	3.000000e+00	6.000000e+00
# price_per_person	244.0	7.888197e+00	2.914234e+00	2.880000e+00	5.800000e+00	7.255000e+00	9.390000e+00	2.027000e+01
# CC Number	244.0	2.563496e+15	2.369340e+15	6.040679e+10	3.040731e+13	3.525318e+15	4.553675e+15	6.596454e+15


'''sort_value()'''

df.sort_values('tip')
# total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID	last_four	Expensive	Tip Quality
# 67	3.07	1.00	Female	Yes	Sat	Dinner	1	3.07	Tiffany Brock	4359488526995267	Sat3455	5267	$	Generous
# 236	12.60	1.00	Male	Yes	Sat	Dinner	2	6.30	Matthew Myers	3543676378973965	Sat5032	3965	$$	Other
# 92	5.75	1.00	Female	Yes	Fri	Dinner	2	2.88	Leah Ramirez	3508911676966392	Fri3780	6392	$	Other
# 111	7.25	1.00	Female	No	Sat	Dinner	1	7.25	Terri Jones	3559221007826887	Sat4801	6887	$	Other
# 0	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410	Sun2959	3410	$$	Other
# ...

# Helpful if you want to reorder after a sort
# https://stackoverflow.com/questions/13148429/how-to-change-the-order-of-dataframe-columns
df.sort_values(['tip','size'])
# 	total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID	last_four	Expensive	Tip Quality
# 67	3.07	1.00	Female	Yes	Sat	Dinner	1	3.07	Tiffany Brock	4359488526995267	Sat3455	5267	$	Generous
# 111	7.25	1.00	Female	No	Sat	Dinner	1	7.25	Terri Jones	3559221007826887	Sat4801	6887	$	Other
# 92	5.75	1.00	Female	Yes	Fri	Dinner	2	2.88	Leah Ramirez	3508911676966392	Fri3780	6392	$	Other
# 236	12.60	1.00	Male	Yes	Sat	Dinner	2	6.30	Matthew Myers	3543676378973965	Sat5032	3965	$$	Other
# 0	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410	Sun2959	3410	$$	Other

'''df.corr() for correlation check'''

df.corr()
# 	total_bill	tip	size	price_per_person	CC Number
# total_bill	1.000000	0.675734	0.598315	0.647554	0.104576
# tip	0.675734	1.000000	0.489299	0.347405	0.110857
# size	0.598315	0.489299	1.000000	-0.175359	-0.030239
# price_per_person	0.647554	0.347405	-0.175359	1.000000	0.135240
# CC Number	0.104576	0.110857	-0.030239	0.135240	1.000000

df[['total_bill','tip']].corr()
# 	total_bill	tip
# total_bill	1.000000	0.675734
# tip	0.675734	1.000000

'''idxmin and idxmax'''

df.head()
# total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID	last_four	Expensive	Tip Quality
# 0	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410	Sun2959	3410	$$	Other
# 1	10.34	1.66	Male	No	Sun	Dinner	3	3.45	Douglas Tucker	4478071379779230	Sun4608	9230	$$	Other
# 2	21.01	3.50	Male	No	Sun	Dinner	3	7.00	Travis Walters	6011812112971322	Sun4458	1322	$$	Other
# 3	23.68	3.31	Male	No	Sun	Dinner	2	11.84	Nathaniel Harris	4676137647685994	Sun5260	5994	$$	Other
# 4	24.59	3.61	Female	No	Sun	Dinner	4	6.15	Tonya Carter	4832732618637221	Sun2251	7221	$$	Other

df['total_bill'].max()
# 50.81

df['total_bill'].idxmax()
# 170

df['total_bill'].idxmin()
# 67

df.iloc[67]
# total_bill                      3.07
# tip                                1
# sex                           Female
# smoker                           Yes
# day                              Sat
# time                          Dinner
# size                               1
# price_per_person                3.07
# Payer Name             Tiffany Brock
# CC Number           4359488526995267
# Payment ID                   Sat3455
# last_four                       5267
# Expensive                          $
# Tip Quality                 Generous
# Name: 67, dtype: object

df.iloc[170]
# total_bill                     50.81
# tip                               10
# sex                             Male
# smoker                           Yes
# day                              Sat
# time                          Dinner
# size                               3
# price_per_person               16.94
# Payer Name             Gregory Clark
# CC Number           5473850968388236
# Payment ID                   Sat1954
# last_four                       8236
# Expensive                        $$$
# Tip Quality                    Other
# Name: 170, dtype: object

'''value_count'''

df['sex'].value_counts()
# Male      157
# Female     87
# Name: sex, dtype: int64

df['Tip Quality'] = df['Tip Quality'].replace(to_replace='Other',value='Ok')
df.head()
# total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID	last_four	Expensive	Tip Quality
# 0	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410	Sun2959	3410	$$	Ok
# 1	10.34	1.66	Male	No	Sun	Dinner	3	3.45	Douglas Tucker	4478071379779230	Sun4608	9230	$$	Ok
# 2	21.01	3.50	Male	No	Sun	Dinner	3	7.00	Travis Walters	6011812112971322	Sun4458	1322	$$	Ok
# 3	23.68	3.31	Male	No	Sun	Dinner	2	11.84	Nathaniel Harris	4676137647685994	Sun5260	5994	$$	Ok
# 4	24.59	3.61	Female	No	Sun	Dinner	4	6.15	Tonya Carter	4832732618637221	Sun2251	7221	$$	Ok

df['sex'] = df['sex'].replace(['Female, Male'],['F','M'])
df.head()
# total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID	last_four	Expensive	Tip Quality
# 0	16.99	1.01	F	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410	Sun2959	3410	$$	Ok
# 1	10.34	1.66	M	No	Sun	Dinner	3	3.45	Douglas Tucker	4478071379779230	Sun4608	9230	$$	Ok
# 2	21.01	3.50	M	No	Sun	Dinner	3	7.00	Travis Walters	6011812112971322	Sun4458	1322	$$	Ok
# 3	23.68	3.31	M	No	Sun	Dinner	2	11.84	Nathaniel Harris	4676137647685994	Sun5260	5994	$$	Ok
# 4	24.59	3.61	F	No	Sun	Dinner	4	6.15	Tonya Carter	4832732618637221	Sun2251	7221	$$	Ok

'''unique'''
df['size'].unique()
# array([2, 3, 4, 1, 6, 5], dtype=int64)

df['size'].nunique()
# 6

df['time'].unique()
# array(['Dinner', 'Lunch'], dtype=object)

'''map'''
# second way for replace value
my_map = {'Female':'F','Male':'M'}
df['sex'] = df['sex'].map(my_map)
df.head()
# total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID	last_four	Expensive	Tip Quality
# 0	16.99	1.01	F	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410	Sun2959	3410	$$	Ok
# 1	10.34	1.66	M	No	Sun	Dinner	3	3.45	Douglas Tucker	4478071379779230	Sun4608	9230	$$	Ok
# 2	21.01	3.50	M	No	Sun	Dinner	3	7.00	Travis Walters	6011812112971322	Sun4458	1322	$$	Ok
# 3	23.68	3.31	M	No	Sun	Dinner	2	11.84	Nathaniel Harris	4676137647685994	Sun5260	5994	$$	Ok
# 4	24.59	3.61	F	No	Sun	Dinner	4	6.15	Tonya Carter	4832732618637221	Sun2251	7221	$$	Ok

'''duplicates'''
df.duplicated() # return true if df contains duplicated rows

simple_df = pd.DataFrame([1,2,2],['a','b','c'])
simple_df
# 	0
# a	1
# b	2
# c	2

simple_df.duplicated()
# a    False
# b    False
# c     True
# dtype: bool

simple_df.drop_duplicates()
# 	0
# a	1
# b	2

'''between'''
df['total_bill'].between(10,20,inclusive=True)

df[df['total_bill'].between(10,20,inclusive=True)]
# total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID	last_four	Expensive	Tip Quality
# 0	16.99	1.01	Female	No	Sun	Dinner	2	8.49	Christy Cunningham	3560325168603410	Sun2959	3410	$$	Ok
# 1	10.34	1.66	Male	No	Sun	Dinner	3	3.45	Douglas Tucker	4478071379779230	Sun4608	9230	$$	Ok
# 8	15.04	1.96	Male	No	Sun	Dinner	2	7.52	Joseph Mcdonald	3522866365840377	Sun6820	0377	$$	Ok
# 9	14.78	3.23	Male	No	Sun	Dinner	2	7.39	Jerome Abbott	3532124519049786	Sun3775	9786	$$	Ok
# 10	10.27	1.71	Male	No	Sun	Dinner	2	5.14	William Riley	566287581219	Sun2546	1219	$$	Ok
# 12	15.42	1.57	Male	No	Sun	Dinner	2	7.71	Chad Harrington	577040572932	Sun1300	2932	$$	Ok
# 13	18.43	3.00	Male	No	Sun	Dinner	4	4.61	Joshua Jones	6011163105616890	Sun2971	6890	$$	Ok
# ...

'''sample'''

df.sample(5) # sample 5 random rows
# total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID	last_four	Expensive	Tip Quality
# 216	28.15	3.00	Male	Yes	Sat	Dinner	5	5.63	Shawn Barnett PhD	4590982568244	Sat7320	8244	$$	Ok
# 136	10.33	2.00	Female	No	Thur	Lunch	2	5.16	Donna Kelly	180048553626376	Thur1393	6376	$$	Ok
# 13	18.43	3.00	Male	No	Sun	Dinner	4	4.61	Joshua Jones	6011163105616890	Sun2971	6890	$$	Ok
# 146	18.64	1.36	Female	No	Thur	Lunch	3	6.21	Kelly Estrada	60463302327	Thur3941	2327	$$	Ok
# 56	38.01	3.00	Male	Yes	Sat	Dinner	4	9.50	James Christensen DDS	349793629453226	Sat8903	3226	$$$	Ok

df.sample(frac=0.1) # sample rows that is total rows * frac (10% of rows from dataframe in this case)
# total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID	last_four	Expensive	Tip Quality
# 73	25.28	5.00	Female	Yes	Sat	Dinner	2	12.64	Julie Holmes	5418689346409571	Sat6065	9571	$$	Ok
# 141	34.30	6.70	Male	No	Thur	Lunch	6	5.72	Steven Carlson	3526515703718508	Thur1025	8508	$$$	Ok
# 239	29.03	5.92	Male	No	Sat	Dinner	3	9.68	Michael Avila	5296068606052842	Sat2657	2842	$$	Ok
# 237	32.83	1.17	Male	Yes	Sat	Dinner	2	16.42	Thomas Brown	4284722681265508	Sat2929	5508	$$$	Ok
# 69	15.01	2.09	Male	Yes	Sat	Dinner	2	7.50	Adam Hall	4700924377057571	Sat855	7571	$$	Ok
# 108	18.24	3.76	Male	No	Sat	Dinner	2	9.12	Steven Grant	4112810433473856	Sat6376	3856	$$	Ok
# 85	34.83	5.17	Female	No	Thur	Lunch	4	8.71	Shawna Cook	6011787464177340	Thur7972	7340	$$$	Ok
# 156	48.17	5.00	Male	No	Sun	Dinner	6	8.03	Ryan Gonzales	3523151482063321	Sun7518	3321	$$$	Ok
# 196	10.34	2.00	Male	Yes	Thur	Lunch	2	5.17	Eric Martin	30442491190342	Thur9862	0342	$$	Ok
# 41	17.46	2.54	Male	No	Sun	Dinner	2	8.73	David Boyer	3536678244278149	Sun9460	8149	$$	Ok
# 236	12.60	1.00	Male	Yes	Sat	Dinner	2	6.30	Matthew Myers	3543676378973965	Sat5032	3965	$$	Ok
# 225	16.27	2.50	Female	Yes	Fri	Lunch	2	8.14	Whitney Arnold	3579111947217428	Fri6665	7428	$$	Ok
# 61	13.81	2.00	Male	Yes	Sat	Dinner	2	6.90	Ryan Hernandez	4766834726806	Sat3030	6806	$$	Ok
# 203	16.40	2.50	Female	Yes	Thur	Lunch	2	8.20	Toni Brooks	3582289985920239	Thur7770	0239	$$	Ok
# 5	25.29	4.71	Male	No	Sun	Dinner	4	6.32	Erik Smith	213140353657882	Sun9679	7882	$$	Ok
# 220	12.16	2.20	Male	Yes	Fri	Lunch	2	6.08	Ricky Johnson	213109508670736	Fri4607	0736	$$	Ok
# 119	24.08	2.92	Female	No	Thur	Lunch	4	6.02	Melanie Jordan	676212062720	Thur8063	2720	$$	Ok
# 96	27.28	4.00	Male	Yes	Fri	Dinner	2	13.64	Eric Carter	4563054452787961	Fri3159	7961	$$	Ok
# 159	16.49	2.00	Male	No	Sun	Dinner	4	4.12	Christopher Soto	30501814271434	Sun1781	1434	$$	Ok
# 26	13.37	2.00	Male	No	Sat	Dinner	2	6.68	Kyle Avery	6531339539615499	Sat6651	5499	$$	Ok
# 129	22.82	2.18	Male	No	Thur	Lunch	3	7.61	Raymond Torres	4855776744024	Thur9424	4024	$$	Ok
# 21	20.29	2.75	Female	No	Sat	Dinner	2	10.14	Natalie Gardner	5448125351489749	Sat9618	9749	$$	Ok
# 94	22.75	3.25	Female	No	Fri	Dinner	2	11.38	Jamie Garza	676318332068	Fri2318	2068	$$	Ok
# 39	31.27	5.00	Male	No	Sat	Dinner	3	10.42	Mr. Brandon Berry	6011525851069856	Sat6373	9856	$$$	Ok


df.nlargest(10,'tip')
# 	total_bill	tip	sex	smoker	day	time	size	price_per_person	Payer Name	CC Number	Payment ID	last_four	Expensive	Tip Quality
# 170	50.81	10.00	Male	Yes	Sat	Dinner	3	16.94	Gregory Clark	5473850968388236	Sat1954	8236	$$$	Ok
# 212	48.33	9.00	Male	No	Sat	Dinner	4	12.08	Alex Williamson	676218815212	Sat4590	5212	$$$	Ok
# 23	39.42	7.58	Male	No	Sat	Dinner	4	9.86	Lance Peterson	3542584061609808	Sat239	9808	$$$	Ok
# 59	48.27	6.73	Male	No	Sat	Dinner	4	12.07	Brian Ortiz	6596453823950595	Sat8139	0595	$$$	Ok
# 141	34.30	6.70	Male	No	Thur	Lunch	6	5.72	Steven Carlson	3526515703718508	Thur1025	8508	$$$	Ok
# 183	23.17	6.50	Male	Yes	Sun	Dinner	4	5.79	Dr. Michael James	4718501859162	Sun6059	9162	$$	Generous
# 214	28.17	6.50	Female	Yes	Sat	Dinner	3	9.39	Marissa Jackson	4922302538691962	Sat3374	1962	$$	Ok
# 47	32.40	6.00	Male	No	Sun	Dinner	4	8.10	James Barnes	3552002592874186	Sun9677	4186	$$$	Ok
# 239	29.03	5.92	Male	No	Sat	Dinner	3	9.68	Michael Avila	5296068606052842	Sat2657	2842	$$	Ok
# 88	24.71	5.85	Male	No	Thur	Lunch	2	12.36	Roger Taylor	4410248629955	Thur9003	9955	$$	Ok
```

## pandas - missing data
```python
"""
A new pd.NA value (singleton) is introduced to represent scalar missing values. Up to now, pandas used several values to represent missing data: np.nan is used for this for float data, np.nan or None for object-dtype data and pd.NaT for datetime-like data. The goal of pd.NA is to provide a “missing” indicator that can be used consistently across data types. pd.NA is currently used by the nullable integer and boolean data types and the new string data type
"""

import numpy as np
import pandas as pd

np.nan
pd.NA
pd.NaT

"""
Note! Typical comparisons should be avoided with Missing Values
> This is generally because the logic here is, since we don't know these values, we can't know if they are equal to each other.
"""
np.nan == np.nan
# False

np.nan in [np.nan]
# True 

np.nan is np.nan
# True

pd.NA == pd.NA
# <NA>

df = pd.read_csv('movie_scores.csv')
df # original dataframe looks like:
# 	first_name	last_name	age	sex	pre_movie_score	post_movie_score
# 0	Tom	Hanks	63.0	m	8.0	10.0
# 1	NaN	NaN	NaN	NaN	NaN	NaN
# 2	Hugh	Jackman	51.0	m	NaN	NaN
# 3	Oprah	Winfrey	66.0	f	6.0	8.0
# 4	Emma	Stone	31.0	f	7.0	9.0

'''Checking and Selecting for Null Values'''

df.isnull()
# 	first_name	last_name	age	sex	pre_movie_score	post_movie_score
# 0	False	False	False	False	False	False
# 1	True	True	True	True	True	True
# 2	False	False	False	False	True	True
# 3	False	False	False	False	False	False
# 4	False	False	False	False	False	False


df.notnull()
# 	first_name	last_name	age	sex	pre_movie_score	post_movie_score
# 0	True	True	True	True	True	True
# 1	False	False	False	False	False	False
# 2	True	True	True	True	False	False
# 3	True	True	True	True	True	True
# 4	True	True	True	True	True	True

df['first_name']
# 0      Tom
# 1      NaN
# 2     Hugh
# 3    Oprah
# 4     Emma
# Name: first_name, dtype: object

df[df['first_name'].notnull()]
# 	first_name	last_name	age	sex	pre_movie_score	post_movie_score
# 0	Tom	Hanks	63.0	m	8.0	10.0
# 2	Hugh	Jackman	51.0	m	NaN	NaN
# 3	Oprah	Winfrey	66.0	f	6.0	8.0
# 4	Emma	Stone	31.0	f	7.0	9.0

df[(df['pre_movie_score'].isnull()) & df['sex'].notnull()]
# first_name	last_name	age	sex	pre_movie_score	post_movie_score
# 2	Hugh	Jackman	51.0	m	NaN	Na

'''drop data'''

# you can use help() for detailed documentaion
help(df.dropna) # method of pandas.core.frame.DataFrame instance Remove missing values.

df.dropna()
# first_name	last_name	age	sex	pre_movie_score	post_movie_score
# 0	Tom	Hanks	63.0	m	8.0	10.0
# 3	Oprah	Winfrey	66.0	f	6.0	8.0
# 4	Emma	Stone	31.0	f	7.0	9.0

df.dropna(thresh=1) # drop all rows that contains null values except for the row that contains at least one not-null value in this case
# > thresh: int, optional; Require that many non-NA values
# first_name	last_name	age	sex	pre_movie_score	post_movie_score
# 0	Tom	Hanks	63.0	m	8.0	10.0
# 2	Hugh	Jackman	51.0	m	NaN	NaN
# 3	Oprah	Winfrey	66.0	f	6.0	8.0
# 4	Emma	Stone	31.0	f	7.0	9.0

df.dropna(axis=1)
# 0
# 1
# 2
# 3
# 4

df.dropna(thresh=4,axis=1)
# 	first_name	last_name	age	sex
# 0	Tom	Hanks	63.0	m
# 1	NaN	NaN	NaN	NaN
# 2	Hugh	Jackman	51.0	m
# 3	Oprah	Winfrey	66.0	f
# 4	Emma	Stone	31.0	f

'''fill data'''

df.fillna('NEW VALUE!')
# 	first_name	last_name	age	sex	pre_movie_score	post_movie_score
# 0	Tom	Hanks	63	m	8	10
# 1	NEW VALUE!	NEW VALUE!	NEW VALUE!	NEW VALUE!	NEW VALUE!	NEW VALUE!
# 2	Hugh	Jackman	51	m	NEW VALUE!	NEW VALUE!
# 3	Oprah	Winfrey	66	f	6	8
# 4	Emma	Stone	31	f	7	9

df['first_name'].fillna("Empty")
# 0      Tom
# 1    Empty
# 2     Hugh
# 3    Oprah
# 4     Emma
# Name: first_name, dtype: object

df['first_name'] = df['first_name'].fillna("Empty")
df
# 	first_name	last_name	age	sex	pre_movie_score	post_movie_score
# 0	Tom	Hanks	63.0	m	8.0	10.0
# 1	Empty	NaN	NaN	NaN	NaN	NaN
# 2	Hugh	Jackman	51.0	m	NaN	NaN
# 3	Oprah	Winfrey	66.0	f	6.0	8.0
# 4	Emma	Stone	31.0	f	7.0	9.0

df['pre_movie_score'].mean()
# 7.0

df['pre_movie_score'].fillna(df['pre_movie_score'].mean())
# 0    8.0
# 1    7.0
# 2    7.0
# 3    6.0
# 4    7.0
# Name: pre_movie_score, dtype: float64

df.fillna(df.mean())
# 	first_name	last_name	age	sex	pre_movie_score	post_movie_score
# 0	Tom	Hanks	63.00	m	8.0	10.0
# 1	Empty	NaN	52.75	NaN	7.0	9.0
# 2	Hugh	Jackman	51.00	m	7.0	9.0
# 3	Oprah	Winfrey	66.00	f	6.0	8.0
# 4	Emma	Stone	31.00	f	7.0	9.0

'''filling with interpolation'''
airline_tix = {'first':100,'business':np.nan,'economy-plus':50,'economy':30}

ser = pd.Series(airline_tix)
ser
# first           100.0
# business          NaN
# economy-plus     50.0
# economy          30.0
# dtype: float64

ser.interpolate()
# first           100.0
# business         75.0
# economy-plus     50.0
# economy          30.0
# dtype: float64

df = pd.DataFrame(ser,columns=['Price'])
df
# 	Price
# first	100.0
# business	NaN
# economy-plus	50.0
# economy	30.0

df.interpolate()
# 	Price
# first	100.0
# business	75.0
# economy-plus	50.0
# economy	30.0

df.interpolate(method='spline',order=2)
# 	index	Price
# 0	first	100.000000
# 1	business	73.333333
# 2	economy-plus	50.000000
# 3	economy	30.000000
```

## pandas - groupby operation & multiindex
```python
import numpy as np
import pandas as pd

df = pd.read_csv('mpg.csv')
df

# mpg	cylinders	displacement	horsepower	weight	acceleration	model_year	origin	name
# 0	18.0	8	307.0	130	3504	12.0	70	1	chevrolet chevelle malibu
# 1	15.0	8	350.0	165	3693	11.5	70	1	buick skylark 320
# 2	18.0	8	318.0	150	3436	11.0	70	1	plymouth satellite
# 3	16.0	8	304.0	150	3433	12.0	70	1	amc rebel sst
# 4	17.0	8	302.0	140	3449	10.5	70	1	ford torino
# ...	...	...	...	...	...	...	...	...	...

'''groupby() method'''
df.groupby('model_year')
# <pandas.core.groupby.generic.DataFrameGroupBy object at 0x00000246790FEC88>

"""Adding an aggregate method call. To use a grouped object, you need to tell pandas how you want to aggregate the data.

Common Options:
    mean(): Compute mean of groups
    sum(): Compute sum of group values
    size(): Compute group sizes
    count(): Compute count of group
    std(): Standard deviation of groups
    var(): Compute variance of groups
    sem(): Standard error of the mean of groups
    describe(): Generates descriptive statistics
    first(): Compute first of group values
    last(): Compute last of group values
    nth() : Take nth value, or a subset if n is a list
    min(): Compute min of group values
    max(): Compute max of group values
"""

# model_year becomes the index! It is NOT a column name,it is now the name of the index
df.groupby('model_year').mean()

# 	mpg	cylinders	displacement	weight	acceleration	origin
# model_year						
# 70	17.689655	6.758621	281.413793	3372.793103	12.948276	1.310345
# 71	21.250000	5.571429	209.750000	2995.428571	15.142857	1.428571
# 72	18.714286	5.821429	218.375000	3237.714286	15.125000	1.535714
# 73	17.100000	6.375000	256.875000	3419.025000	14.312500	1.375000
# 74	22.703704	5.259259	171.740741	2877.925926	16.203704	1.666667
# 75	20.266667	5.600000	205.533333	3176.800000	16.050000	1.466667
# 76	21.573529	5.647059	197.794118	3078.735294	15.941176	1.470588
# 77	23.375000	5.464286	191.392857	2997.357143	15.435714	1.571429
# 78	24.061111	5.361111	177.805556	2861.805556	15.805556	1.611111
# 79	25.093103	5.827586	206.689655	3055.344828	15.813793	1.275862
# 80	33.696552	4.137931	115.827586	2436.655172	16.934483	2.206897
# 81	30.334483	4.620690	135.310345	2522.931034	16.306897	1.965517
# 82	31.709677	4.193548	128.870968	2453.548387	16.638710	1.645161

avg_year = df.groupby('model_year').mean()
avg_year.index
# Int64Index([70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82], dtype='int64', name='model_year')

avg_year.columns
# Index(['mpg', 'cylinders', 'displacement', 'weight', 'acceleration', 'origin'], dtype='object')

avg_year['mpg']
# model_year
# 70    17.689655
# 71    21.250000
# 72    18.714286
# 73    17.100000
# 74    22.703704
# 75    20.266667
# 76    21.573529
# 77    23.375000
# 78    24.061111
# 79    25.093103
# 80    33.696552
# 81    30.334483
# 82    31.709677
# Name: mpg, dtype: float64

df.groupby('model_year').mean()['mpg']
# model_year
# 70    17.689655
# 71    21.250000
# 72    18.714286
# 73    17.100000
# 74    22.703704
# 75    20.266667
# 76    21.573529
# 77    23.375000
# 78    24.061111
# 79    25.093103
# 80    33.696552
# 81    30.334483
# 82    31.709677
# Name: mpg, dtype: float64

df.groupby('model_year').describe()
df.groupby('model_year').describe().transpose()
# same as before

'''group by multiple Columns'''
df.groupby(['model_year','cylinders']).mean()
# 		mpg	displacement	weight	acceleration	origin
# model_year	cylinders					
# 70	4	25.285714	107.000000	2292.571429	16.000000	2.285714
#       6	20.500000	199.000000	2710.500000	15.500000	1.000000
#       8	14.111111	367.555556	3940.055556	11.194444	1.000000
# 71	4	27.461538	101.846154	2056.384615	16.961538	1.923077
#       6	18.000000	243.375000	3171.875000	14.750000	1.000000
#       8	13.428571	371.714286	4537.714286	12.214286	1.000000
# 72	3	19.000000	70.000000	2330.000000	13.500000	3.000000
#       4	23.428571	111.535714	2382.642857	17.214286	1.928571
#       8	13.615385	344.846154	4228.384615	13.000000	1.000000

df.groupby(['model_year','cylinders']).mean().index
# MultiIndex([(70, 4),
#             (70, 6),
#             (70, 8),
#             (71, 4),
#             (71, 6),
#             (71, 8),
#             (72, 3),
#             (72, 4),
#             (72, 8),
#             (73, 3),
#             (73, 4),
#             (73, 6),
#             (73, 8),
#             (74, 4),
#             (74, 6),
#             (74, 8),
#             (75, 4),
#             (75, 6),
#             (75, 8),
#             (76, 4),
#             (76, 6),
#             (76, 8),
#             (77, 3),
#             (77, 4),
#             (77, 6),
# ...
#             (81, 6),
#             (81, 8),
#             (82, 4),
#             (82, 6)],
#            names=['model_year', 'cylinders'])

year_cyl = df.groupby(['model_year','cylinders']).mean()
year_cyl
# same as above

year_cyl.index
# same as above df.groupby(['model_year','cylinders']).mean().index

year_cyl.index.levels
# FrozenList([[70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82], [3, 4, 5, 6, 8]])

year_cyl.index.names
# FrozenList(['model_year', 'cylinders'])

'''indexing with the hierarchical index'''
year_cyl.head()
# 	mpg	displacement	weight	acceleration	origin
# model_year	cylinders					
# 70	4	25.285714	107.000000	2292.571429	16.000000	2.285714
#       6	20.500000	199.000000	2710.500000	15.500000	1.000000
#       8	14.111111	367.555556	3940.055556	11.194444	1.000000
# 71	4	27.461538	101.846154	2056.384615	16.961538	1.923077
#       6	18.000000	243.375000	3171.875000	14.750000	1.000000

year_cyl.loc[70]
# 	mpg	displacement	weight	acceleration	origin
# cylinders					
# 4	25.285714	107.000000	2292.571429	16.000000	2.285714
# 6	20.500000	199.000000	2710.500000	15.500000	1.000000
# 8	14.111111	367.555556	3940.055556	11.194444	1.000000

year_cyl.loc[[70,72]]
# 	mpg	displacement	weight	acceleration	origin
# model_year	cylinders					
# 70	4	25.285714	107.000000	2292.571429	16.000000	2.285714
#       6	20.500000	199.000000	2710.500000	15.500000	1.000000
#       8	14.111111	367.555556	3940.055556	11.194444	1.000000
# 71	4	27.461538	101.846154	2056.384615	16.961538	1.923077
#       6	18.000000	243.375000	3171.875000	14.750000	1.000000

# grab a single row
year_cyl.loc[(70,8)]
# mpg               14.111111
# displacement     367.555556
# weight          3940.055556
# acceleration      11.194444
# origin             1.000000
# Name: (70, 8), dtype: float64

'''grab based on cross-secion with .xs()'''

"""
This method takes a `key` argument to select data at a particular
level of a MultiIndex.

Parameters
----------
    key : label or tuple of label
        Label contained in the index, or partially in a MultiIndex.
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Axis to retrieve cross-section on.
    level : object, defaults to first n levels (n=1 or len(key))
        In case of a key partially contained in a MultiIndex, indicate
        which levels are used. Levels can be referred by label or position.
"""

year_cyl.xs(key=70,axis=0,level='model_year')
#       mpg	displacement	weight	acceleration	origin
# cylinders					
# 4	25.285714	107.000000	2292.571429	16.000000	2.285714
# 6	20.500000	199.000000	2710.500000	15.500000	1.000000
# 8	14.111111	367.555556	3940.055556	11.194444	1.000000

# Mean column values for 4 cylinders per year
year_cyl.xs(key=4,axis=0,level='cylinders')
# 	mpg	displacement	weight	acceleration	origin
# model_year					
# 70	25.285714	107.000000	2292.571429	16.000000	2.285714
# 71	27.461538	101.846154	2056.384615	16.961538	1.923077
# 72	23.428571	111.535714	2382.642857	17.214286	1.928571
# 73	22.727273	109.272727	2338.090909	17.136364	2.000000
# 74	27.800000	96.533333	2151.466667	16.400000	2.200000
# 75	25.250000	114.833333	2489.250000	15.833333	2.166667
# 76	26.766667	106.333333	2306.600000	16.866667	1.866667
# 77	29.107143	106.500000	2205.071429	16.064286	1.857143
# 78	29.576471	112.117647	2296.764706	16.282353	2.117647
# 79	31.525000	113.583333	2357.583333	15.991667	1.583333
# 80	34.612000	111.000000	2360.080000	17.144000	2.200000
# 81	32.814286	108.857143	2275.476190	16.466667	2.095238
# 82	32.071429	118.571429	2402.321429	16.703571	1.714286


df[df['cylinders'].isin([6,8])].groupby(['model_year','cylinders']).mean()
# 		mpg	displacement	weight	acceleration	origin
# model_year	cylinders					
# 70	6	20.500000	199.000000	2710.500000	15.500000	1.000000
#     8	14.111111	367.555556	3940.055556	11.194444	1.000000
# 71	6	18.000000	243.375000	3171.875000	14.750000	1.000000
#     8	13.428571	371.714286	4537.714286	12.214286	1.000000
# 72	8	13.615385	344.846154	4228.384615	13.000000	1.000000
# 73	6	19.000000	212.250000	2917.125000	15.687500	1.250000
#     8	13.200000	365.250000	4279.050000	12.250000	1.000000
# 74	6	17.857143	230.428571	3320.000000	16.857143	1.000000
#     8	14.200000	315.200000	4438.400000	14.700000	1.000000
# 75	6	17.583333	233.750000	3398.333333	17.708333	1.000000
#     8	15.666667	330.500000	4108.833333	13.166667	1.000000
# 76	6	20.000000	221.400000	3349.600000	17.000000	1.300000
#     8	14.666667	324.000000	4064.666667	13.222222	1.000000
# 77	6	19.500000	220.400000	3383.000000	16.900000	1.400000
#     8	16.000000	335.750000	4177.500000	13.662500	1.000000
# 78	6	19.066667	213.250000	3314.166667	16.391667	1.166667
#     8	19.050000	300.833333	3563.333333	13.266667	1.000000
# 79	6	22.950000	205.666667	3025.833333	15.433333	1.000000
#     8	18.630000	321.400000	3862.900000	15.400000	1.000000
# 80	6	25.900000	196.500000	3145.500000	15.050000	2.000000
# 81	6	23.428571	184.000000	3093.571429	15.442857	1.714286
#     8	26.600000	350.000000	3725.000000	19.000000	1.000000
# 82	6	28.333333	225.000000	2931.666667	16.033333	1.000000

'''swap levels'''
year_cyl.swaplevel().head()
# 		mpg	displacement	weight	acceleration	origin
# cylinders	model_year					
# 4	70	25.285714	107.000000	2292.571429	16.000000	2.285714
# 6	70	20.500000	199.000000	2710.500000	15.500000	1.000000
# 8	70	14.111111	367.555556	3940.055556	11.194444	1.000000
# 4	71	27.461538	101.846154	2056.384615	16.961538	1.923077
# 6	71	18.000000	243.375000	3171.875000	14.750000	1.000000

'''sort multiindex'''
year_cyl.sort_index(level='model_year',ascending=False)
# 		mpg	displacement	weight	acceleration	origin
# model_year	cylinders					
# 82	6	28.333333	225.000000	2931.666667	16.033333	1.000000
#     4	32.071429	118.571429	2402.321429	16.703571	1.714286
# 81	8	26.600000	350.000000	3725.000000	19.000000	1.000000
#     6	23.428571	184.000000	3093.571429	15.442857	1.714286
#     4	32.814286	108.857143	2275.476190	16.466667	2.095238
# 80	6	25.900000	196.500000	3145.500000	15.050000	2.000000
#     5	36.400000	121.000000	2950.000000	19.900000	2.000000
#     4	34.612000	111.000000	2360.080000	17.144000	2.200000
#     3	23.700000	70.000000	2420.000000	12.500000	3.000000
# 79	8	18.630000	321.400000	3862.900000	15.400000	1.000000
#     6	22.950000	205.666667	3025.833333	15.433333	1.000000
#     5	25.400000	183.000000	3530.000000	20.100000	2.000000
#     4	31.525000	113.583333	2357.583333	15.991667	1.583333
# ... ... ...

year_cyl.sort_index(level='cylinders',ascending=False)
# 		mpg	displacement	weight	acceleration	origin
# model_year	cylinders					
# 81	8	26.600000	350.000000	3725.000000	19.000000	1.000000
# 79	8	18.630000	321.400000	3862.900000	15.400000	1.000000
# 78	8	19.050000	300.833333	3563.333333	13.266667	1.000000
# 77	8	16.000000	335.750000	4177.500000	13.662500	1.000000
# 76	8	14.666667	324.000000	4064.666667	13.222222	1.000000
# 75	8	15.666667	330.500000	4108.833333	13.166667	1.000000
# 74	8	14.200000	315.200000	4438.400000	14.700000	1.000000
# 73	8	13.200000	365.250000	4279.050000	12.250000	1.000000
# 72	8	13.615385	344.846154	4228.384615	13.000000	1.000000
# 71	8	13.428571	371.714286	4537.714286	12.214286	1.000000
# 70	8	14.111111	367.555556	3940.055556	11.194444	1.000000
# 82	6	28.333333	225.000000	2931.666667	16.033333	1.000000
# 81	6	23.428571	184.000000	3093.571429	15.442857	1.714286
# 80	6	25.900000	196.500000	3145.500000	15.050000	2.000000
# 79	6	22.950000	205.666667	3025.833333	15.433333	1.000000
# 78	6	19.066667	213.250000	3314.166667	16.391667	1.166667

'''advanced: agg() method'''
df.agg(['median','mean'])
#           mpg	    cylinders	displacement	weight	acceleration	model_year	origin
# median	23.000000	4.000000	148.500000	2803.500000	15.50000	76.00000	1.000000
# mean	23.514573	5.454774	193.425879	2970.424623	15.56809	76.01005	1.572864

df.agg(['sum','mean'])[['mpg','weight']]
# 	mpg	weight
# sum	9358.800000	1.182229e+06
# mean	23.514573	2.970425e+03

'''specify aggregate methods per column'''
"""
**agg()** is very powerful,allowing you to pass in a dictionary where the keys are the columns and the values are a list of aggregate methods.
"""
df.agg({'mpg':['median','mean'],'weight':['mean','std']})
#           mpg	        weight
# mean	    23.514573	2970.424623
# median	23.000000	NaN
# std	    NaN	        846.841774

'''agg() with groupby()'''
df.groupby('model_year').agg({'mpg':['median','mean'],'weight':['mean','std']})
# 	mpg	                weight
#     median	mean	    mean	    std
# model_year				
# 70	16.00	17.689655	3372.793103	852.868663
# 71	19.00	21.250000	2995.428571	1061.830859
# 72	18.50	18.714286	3237.714286	974.520960
# 73	16.00	17.100000	3419.025000	974.809133
# 74	24.00	22.703704	2877.925926	949.308571
# 75	19.50	20.266667	3176.800000	765.179781
# 76	21.00	21.573529	3078.735294	821.371481
# 77	21.75	23.375000	2997.357143	912.825902
# 78	20.70	24.061111	2861.805556	626.023907
# 79	23.90	25.093103	3055.344828	747.881497
# 80	32.70	33.696552	2436.655172	432.235491
# 81	31.60	30.334483	2522.931034	533.600501
# 82	32.00	31.709677	2453.548387	354.276713

```
- [group by](https://pandas.pydata.org/docs/reference/groupby.html)
- [swapping levels](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#swapping-levels-with-swaplevel)
- [reorder_levels](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#reordering-levels-with-reorder-levels)
- [sorting multiindex](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#sorting-a-multiindex)

## pandas - combining dataframes(concatenation, inner merge, left and right merge, and outer merge)
```python
import numpy as np
import pandas as pd

data_one = {'A': ['A0', 'A1', 'A2', 'A3'],'B': ['B0', 'B1', 'B2', 'B3']}
data_two = {'C': ['C0', 'C1', 'C2', 'C3'], 'D': ['D0', 'D1', 'D2', 'D3']}
one = pd.DataFrame(data_one)
two = pd.DataFrame(data_two)

one
#     A	B
# 0	A0	B0
# 1	A1	B1
# 2	A2	B2
# 3	A3	B3

two
#   C	D
# 0	C0	D0
# 1	C1	D1
# 2	C2	D2
# 3	C3	D3

'''concate along rows'''
axis0 = pd.concat([one,two],axis=0)
axis0
#   A	B	C	D
# 0	A0	B0	NaN	NaN
# 1	A1	B1	NaN	NaN
# 2	A2	B2	NaN	NaN
# 3	A3	B3	NaN	NaN
# 0	NaN	NaN	C0	D0
# 1	NaN	NaN	C1	D1
# 2	NaN	NaN	C2	D2
# 3	NaN	NaN	C3	D3

'''concate along column'''
axis1 = pd.concat([one,two],axis=1)
axis1
#   A	B	C	D
# 0	A0	B0	C0	D0
# 1	A1	B1	C1	D1
# 2	A2	B2	C2	D2
# 3	A3	B3	C3	D3

two.columns = one.columns # this rename the column name
pd.concat([one,two], axis=0)
# 	A	B
# 0	A0	B0
# 1	A1	B1
# 2	A2	B2
# 3	A3	B3
# 0	C0	D0
# 1	C1	D1
# 2	C2	D2
# 3	C3	D3

'''merge'''
registrations = pd.DataFrame({'reg_id':[1,2,3,4],'name':['Andrew','Bobo','Claire','David']})
logins = pd.DataFrame({'log_id':[1,2,3,4],'name':['Xavier','Andrew','Yolanda','Bobo']})

registrations
# 	reg_id	name
# 0	1	Andrew
# 1	2	Bobo
# 2	3	Claire
# 3	4	David

logins
#    log_id	name
# 0	1	Xavier
# 1	2	Andrew
# 2	3	Yolanda
# 3	4	Bobo

'''inner, left, right, and outer joins'''
# Notice pd.merge doesn't take in a list like concat
pd.merge(registrations,logins,how='inner',on='name')
#   reg_id    name    log_id
# 0	1	      Andrew  2
# 1	2	      Bobo	  4

# Pandas smart enough to figure out key column (on parameter) if only one column name matches up
pd.merge(registrations,logins,how='inner')
# reg_id	name	log_id
# 0	1	Andrew	2
# 1	2	Bobo	4

pd.merge(registrations,logins,how='left')
# reg_id	name	log_id
# 0	1	Andrew	2.0
# 1	2	Bobo	4.0
# 2	3	Claire	NaN
# 3	4	David	NaN

pd.merge(registrations,logins,how='right')
# reg_id	name	log_id
# 0	1.0	Andrew	2
# 1	2.0	Bobo	4
# 2	NaN	Xavier	1
# 3	NaN	Yolanda	3

pd.merge(registrations,logins,how='outer')
# 	reg_id	name	log_id
# 0	1.0	Andrew	2.0
# 1	2.0	Bobo	4.0
# 2	3.0	Claire	NaN
# 3	4.0	David	NaN
# 4	NaN	Xavier	1.0
# 5	NaN	Yolanda	3.0

'''join on index or column'''
registrations = registrations.set_index("name")
registrations
# 	reg_id
# name	
# Andrew	1
# Bobo	2
# Claire	3
# David	4

"""
**Use combinations of left_on,right_on,left_index,right_index to merge a column or index on each other**
"""
pd.merge(registrations,logins,left_index=True,right_on='name')
# reg_id log_id	name
# 1	1	2	Andrew
# 3	2	4	Bobo

pd.merge(logins,registrations,right_index=True,left_on='name')
# 	log_id	name	reg_id
# 1	2	Andrew	1
# 3	4	Bobo	2

'''dealing weith different key column names in joined tables'''
registrations = registrations.reset_index()
registrations
#   name	reg_id
# 0	Andrew	1
# 1	Bobo	2
# 2	Claire	3
# 3	David	4

logins
#   log_id	name
# 0	1	Xavier
# 1	2	Andrew
# 2	3	Yolanda
# 3	4	Bobo

registrations.columns = ['reg_name','reg_id']
registrations
#     reg_name	reg_id
# 0	Andrew	1
# 1	Bobo	2
# 2	Claire	3
# 3	David	4

pd.merge(registrations,logins,left_on='reg_name',right_on='name')
# 	reg_name	reg_id	log_id	name
# 0	Andrew	1	2	Andrew
# 1	Bobo	2	4	Bobo

pd.merge(registrations,logins,left_on='reg_name',right_on='name').drop('reg_name',axis=1)
# 	reg_id	log_id	name
# 0	1	2	Andrew
# 1	2	4	Bobo

'''pandas automatically tags duplicate columns'''
registrations.columns = ['name','id']
logins.columns = ['id','name']

registrations
#     name	id
# 0	Andrew	1
# 1	Bobo	2
# 2	Claire	3
# 3	David	4

logins
#     id	name
# 0	1	Xavier
# 1	2	Andrew
# 2	3	Yolanda
# 3	4	Bobo

# _x is for left
# _y is for right
pd.merge(registrations,logins,on='name')
#   name	id_x	id_y
# 0	Andrew	1	    2
# 1	Bobo	2	    4

pd.merge(registrations,logins,on='name',suffixes=('_reg','_log'))
#     name	id_reg	id_log
# 0	Andrew	1	    2
# 1	Bobo	2	    4


```