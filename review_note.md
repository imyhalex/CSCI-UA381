# Bash

# Python

## String
```python
x='        hello everyone... today is Thursday.       '
x.lower()
x.title()
x.upper()
x.lstrip()
x.strip()
x.swapcase()
x.find('is')
x.count('is')

# Outputs
#         hello everyone... today is thursday.       
#         Hello Everyone... Today Is Thursday.       
#         HELLO EVERYONE... TODAY IS THURSDAY.       
# hello everyone... today is Thursday.       
# hello everyone... today is Thursday.
#         HELLO EVERYONE... TODAY IS tHURSDAY.       
# 32

```

## List
```python
L1=[1,2,3]
L2=['welcome','hi', False]
L1+L2 # [1,2,3,'welcome','hi',False]
'welcome' in L1 # False
L2[-1] # False
```

> Use deep copy when want to copy the whole list (includ nested list)
```python
# Copy
L1=[1,2,3]
L2=[4,5]
L3=[L1, L2]
L4=L3

L1[0] = 500

L3 # [[500, 2, 3], [4, 5]]

L4 # [[500, 2, 3], [4, 5]]

L3[1] = 7000
L3 # [[500, 2, 3], 7000]

L4 # [[500, 2, 3], 7000]

##############################################################
import copy
L5=copy.deepcopy(L4) # it will create a new L5, not reference
##############################################################

# Shallow copy for L1
L1=[1,2,3]
L2=[4,5, L1]
L1[0]=100
print(L1)
print(L2)
# Outputs
# [100, 2, 3]
# [4, 5, [100, 2, 3]]

# Deep Copy for L1
L1=[1,2,3]
L2=[4,5, list(L1)]
L1[0]=100
print(L1)
print(L2)
# Outputs
# [100, 2, 3]
# [4, 5, [1, 2, 3]]

# Shallow copy for L1 and L2
L1=[1,2,3]
L2=[4,5, L1]
L3=[6,7,L2]  # L3  --> [6,7, ref_to_L2 ]
L1[0]=100
print(L1)
print(L2)
print(L3)
# Outputs
# [100, 2, 3]
# [4, 5, [100, 2, 3]]
# [6, 7, [4, 5, [100, 2, 3]]]

# Deep Copy for L2 but shallow copy for L1
L1=[1,2,3]
L2=[4,5, L1]
L3=[6,7,list(L2)]   # L3  --> [6,7, [4,5, ref_to_L1]]
L1[0]=100
print(L1)
print(L2)
print(L3)
# Outputs
# [100, 2, 3]
# [4, 5, [100, 2, 3]]
# [6, 7, [4, 5, [100, 2, 3]]]

# Deep copy
import copy
L1=[1,2,3]
L2=[4,5, L1]
L3=[6,7,copy.deepcopy(L2)]   # L3  --> [6,7, [4,5, [1,2,3]]]
L1[0]=100
print(L1)
print(L2)
print(L3)
# Outputs
# [100, 2, 3]
# [4, 5, [100, 2, 3]]
# [6, 7, [4, 5, [1, 2, 3]]]

# Deep copy
L1=[1,2,3]
L2=[4,5, list(L1)]
L3=[6,7,list(L2)]   # L3  --> [6,7, [4,5, [1,2,3]]]
L1[0]=100
print(L1)
print(L2)
print(L3)
# Outputs
# [100, 2, 3]
# [4, 5, [1, 2, 3]]
# [6, 7, [4, 5, [1, 2, 3]]]
```

## Tuple
 - is similar to list but (immutable)
 - tuple can contains any data type

```python
#  create tuple
t=tuple()
t=()
t=(1,3,4,5,6)

# tuple can contain any type of items
t=(1,2,3,True, 4.5, "welcome")
t # (1, 2, 3, True, 4.5, 'welcome')

# Access items (indexing)
t[-1] # welcome
t[3] # True

# Slicing
# t[start: end: step]
t[1:5] # (2, 3, True, 4.5)
t[:5:2] # (1, 3, 4.5)

# Operators
# +, *, in, len, count, index
t1=(1,2,3)
t2=(4,5, False)
t1+t2
t1*4
4 in t1
# Outputs
# (1, 2, 3, 4, 5, False)
# (1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3)
# False

# lists are mutable
print(L)
L[0]=100
print(L)
# Outputs
# [100, 2, 4, 5, 10, True, 4.54]
# [100, 2, 4, 5, 10, True, 4.54]

# tuples are immutable
print(t)
# t[0]=100    # NOT POSSIBLE
# print(t) -> (1, 2, 3, True, 4.5, 'welcome')

# Nested tuples
t1=(1,2,3)
t2=(4,5,"Welcome")
t3=(t1, "thank you", t2)
```

## Set
 - set stores one instance of the value only (it doesn't store duplicates)
```python
# create sets
s=set()
s={1,2,4,5}    # this is ok to create set
# s={}           # this is NOT ok to create set (this will create dict)
type(s) # set

# set stores one instance of the value only (it doesn't store duplicates)
s={1,2,3,1,1,1,1, "hasan", 'james', "hasan", 1, "hasan"}
print(s) # {1, 2, 'james', 3, 'hasan'}

# operators that we can use with set
# in, &, |, -, ^
print('hasan' in s) # True

# suppose we have two sets
s1={'hasan', 'alma', 'sara', 'mike'}
s2={'sara', 'james', 'alma'}

'''here are some examples of operators'''
# and (what is shared between the two sets)
print(s1 & s2)
print(s1.intersection(s2))
print(s2.intersection(s1))
# Output
# {'sara', 'alma'}
# {'sara', 'alma'}
# {'sara', 'alma'}

# or (what is contained in both s1 and s2 , i.e. union)
s1 | s2
s1.union(s2)
s2.union(s1)
# Output
# {'alma', 'hasan', 'james', 'mike', 'sara'}

# difference (what is contained in s1 NOT in s2)
s1-s2
s1.difference(s2)
# Output
# {'hasan', 'mike'}

s2-s1
s2.difference(s1)
# Output
# {'james'}

# symmetric difference (what is NOT shared between s1 and s2, i.e. (s1-s2)|(s2-s1) )
s1^s2
s2^s1
s1.symmetric_difference(s2)
s2.symmetric_difference(s1)
# Output
# {'hasan', 'james', 'mike'}
```

## Dict
 - dictionaries are mutable
```python
# create dictionary
d=dict()
d={}
d

salaries={'james':2000, 'sara':3000,'maya':5000}
salaries['sara']=6000

print(salaries.values())
print(salaries.keys())
print(salaries.items())
# Outputs
# dict_values([2000, 6000, 5000])
# dict_keys(['james', 'sara', 'maya'])
# dict_items([('james', 2000), ('sara', 6000), ('maya', 5000)])

print(list(salaries.values())[0])
print(list(salaries.keys())[0])
list(salaries.items())[0]
# Output
# 2000
# james
# ('james', 2000)

# dictionary's values can be anything
salaries['maya'] = 'Five Thousands' 
salaries
# Output
# {'james': 2000, 'sara': 6000, 'maya': 'Five Thousands'}

salaries['maya']=[200,400, 1500, 700]
salaries
# Output
# {'james': 2000, 'sara': 6000, 'maya': [200, 400, 1500, 700]}

salaries['james'] = {'jan': 200, 'may':500}
salaries
# Output
# {'james': {'jan': 200, 'may': 500},
#  'sara': 6000,
#  'maya': [200, 400, 1500, 700]}

# dictionay's keys can be any immutable thing
salaries['hasan']=5000
salaries[('mike','robert')]= 6000
salaries[112] = 7000
salaries[('tara',1995)] = 5500
salaries
# Output
# {'james': {'jan': 200, 'may': 500},
#  'sara': 6000,
#  'maya': [200, 400, 1500, 700],
#  'hasan': 5000,
#  ('mike', 'robert'): 6000,
#  112: 7000,
#  ('tara', 1995): 5500}

salaries[['mike','smith']]=2000    # Does't work

'''Operators, functions and methods'''
# in   (is used to verify if exist in keys)
# 5500 in salaries         # 5500 in salaries.keys()
# 'hasan' in salaries      #'hasan' in salaries.keys()

# del
del salaries['hasan']
```

## Convert between data structures
```python
# list(), tuple(), set(), dict()
L=[1,2,3]
tuple(L)
# Output
# (1, 2, 3)

t=(10,20,30)
list(t)
# Output
# [10, 20, 30]

L=[1,2,3,2,3,3,3]
set(L)
# Output
# {1, 2, 3}

L=[1,2,3]
# dict(L)    # Not possible

# convert list of tuples into dict
LL= [('hasan',40), ('sara',20), ('william', [10,20,30])]
d=dict(LL)
d
# Output
# {'hasan': 40, 'sara': 20, 'william': [10, 20, 30]}

# convert dict into list of tuples
list(d.items())
# Output
# [('hasan', 40), ('sara', 20), ('william', [10, 20, 30])]

print(list(d))
print(list(d.keys()))
print(list(d.values()))
print(list(d.items()))
# Output
# ['hasan', 'sara', 'william']
# ['hasan', 'sara', 'william']
# [40, 20, [10, 20, 30]]
# [('hasan', 40), ('sara', 20), ('william', [10, 20, 30])]

print(tuple(d))
print(tuple(d.keys()))
print(tuple(d.values()))
print(tuple(d.items()))
# Output
# ('hasan', 'sara', 'william')
# ('hasan', 'sara', 'william')
# (40, 20, [10, 20, 30])
# (('hasan', 40), ('sara', 20), ('william', [10, 20, 30]))

print(set(d))
print(set(d.keys()))
# print(set(d.values()))
# print(set(d.items()))
# Output
# {'william', 'sara', 'hasan'}
# {'william', 'sara', 'hasan'}
```
## Some important functions
```python
# new we have these data structures
print(L)
print(t)
print(s)
print(d)
# Output
# [1, 2, 3]
# (10, 20, 30)
# {1, 2, 3}
# {'hasan': 40, 'sara': 20, 'william': [10, 20, 30]}

# len, sum, max, min, zip
print(len(L))
print(len(t))
print(len(s))
print(len(d))   # len(d.keys())
# Output
# 3
# 3
# 3
# 3

#  using the function zip
L1=[1,2,3]
L2=['james','sara','alma','william']
L3=[2000,3000,5000,2500]
L=list(zip(L1,L2, L3))
L
# Output
# [(1, 'james', 2000), (2, 'sara', 3000), (3, 'alma', 5000)]

# unzip using the function zip
print(list(zip(*L)))
list(list(zip(*L))[0])
# Output
# [(1, 2, 3), ('james', 'sara', 'alma'), (2000, 3000, 5000)]
# [1, 2, 3]

x1,x2,x3= list(zip(*L))
# This will be:
# x1
# (1, 2, 3)
# x2
# ('james', 'sara', 'alma')
# x3
# (2000, 3000, 5000)

L=[1,2,3]
dict(list(zip(L,L)))
# Output
# {1: 1, 2: 2, 3: 3}
```

## function in python
```python

def myfun(x,y=0):
    return x+y, x-y, x*y

result=myfun(10,5) # the result -> (15, 5, 50)
type(result) # tuple

myfun(10) # (10, 10, 0)

a,s,m = myfun(10,5)
# Output
# a is 15
# s is 5
# m is 50
```

## some important functions
 - len
 - min
 - max
 - sum
 - sorted

```python
'''sorted()'''
names=['hasan', 'sara', 'robert', 'william', 'anas', 'sam']
sorted(names)
# ['anas', 'hasan', 'robert', 'sam', 'sara', 'william']

sorted(names, key=len)
# ['sam', 'sara', 'anas', 'hasan', 'robert', 'william']

def get_last_char(n):
    return n[-1]

'''sorted(iterable, key, reverse=True/False)'''
sorted(names, key=get_last_char)
# ['sara', 'william', 'sam', 'hasan', 'anas', 'robert']

'''some concepts of lambda functions'''
get_last=lambda n:n[-1]
get_last('james')
# 's'

my_lambda=lambda x,y: (x+y, x-y,x*y)
my_lambda(10,5)
# (15, 5, 50)
```

## utilizing map with functions
```python
names # this contains -> ['hasan', 'sara', 'robert', 'william', 'anas', 'sam']

'''map(function, iterable)'''
list(map( len, names))
# [5, 4, 6, 7, 4, 3]

list(map(get_last_char, names))
# ['n', 'a', 't', 'm', 's', 'm']

list(map(lambda n:n[-1],names))
# ['n', 'a', 't', 'm', 's', 'm']

'''Example - 1'''
employees = [('hasan', 40, 5000), ('sara', 20, 6000), ('alma', 25, 7000)]

# write one line code to sort this employees by age
sorted(employees, key=lambda x:x[1])

# write one line code that will find the total sum of all employees salaries
sum(map(lambda x:x[2], employees))
```

## filter
```python
names # this contains value -> ['hasan', 'sara', 'robert', 'william', 'anas', 'sam']

'''filter(function, iterable)'''
list(filter(lambda n:len(n)>3 ,names))
# ['hasan', 'sara', 'robert', 'william', 'anas']

employees # it contains -> [('hasan', 40, 5000), ('sara', 20, 6000), ('alma', 25, 7000)]

# filter all employees based on age (age <30)
list(filter(lambda e:e[1]<30,employees))
# [('sara', 20, 6000), ('alma', 25, 7000)]

'''Example - 2'''
# write one line code to find names of employees with salary less than 7000
list(map(lambda x:x[0] ,filter(lambda x:x[2] < 7000, employees)))
```

## iterators
```python
names # it contains -> ['hasan', 'sara', 'robert', 'william', 'anas', 'sam']

i=iter(names)
type(i) # list_iterator

next(i) # give you one element from list for each execution

list(i) # convert it back to list
```

## Control Statements
```python
# if, if else, if elif else, nested if
# while, for, break, continue

'''list comprehension'''
s=[]
for e in employees:
    s.append(e[1])
s
# [20, 23, 40]

[e[1] for e in employees]
# [20, 23, 40]

list(map(lambda e:e[1], employees))
# [20, 23, 40]

names # it contains -> ['hasan', 'sara', 'alma']

new_names=[]
for n in names:
    new_names.append(n.upper())
new_names
# ['HASAN', 'SARA', 'ALMA']
# this is same as
[n.upper() for n in names]
# or 
list(map(lambda n:n.upper(), names))

'''example - 2'''
employees # it contains -> [('james', 20, 2000), ('sara', 23, 2500), ('alma', 40, 1500)]

# filter-like
[e for e in employees if e[1]>20]
# [('sara', 23, 2500), ('alma', 40, 1500)]

# lambda(filter))-like
[e[0] for e in employees if e[1]>20]
# ['sara', 'alma']

# it names -> ['hasan', 'sara', 'alma']
# conditional statment has to be front if 'else' contains
[n if n[0]=='h' else n.upper() for n in names]
# ['hasan', 'SARA', 'ALMA']
```

## generators
> when we enconter data that is too big
```python
'''a huge list'''
L=list(range(1000_000))

def g1(list_of_numbers):
    for n in list_of_numbers:
        yield n*n

# displat execution time
%%timeit
g1_result=g1(L)
# 190 ns ± 6.97 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)

next(g1_result) # take out one value per execution

# [v*v for v in L]  # return list
[v*v for v in L].__sizeof__()
# 8448712

# (v*v for v in L)    # returns generator
(v*v for v in L).__sizeof__()
# 192

map(lambda v:v*v, L)
# <map at 0x7f8a386a78e0>
```

## Exception
```python
names=['james','sarah','mike','hasan','tara','william']
for n in names:
    if n=='hasan':
        raise Exception('hasan is not welcomed')
    print('hi..', n)
#######################
hi.. james
hi.. sarah
hi.. mike
---------------------------------------------------------------------------
Exception                                 Traceback (most recent call last)
Cell In[45], line 3
      1 for n in names:
      2     if n=='hasan':
----> 3         raise Exception('hasan is not welcomed')
      4     print('hi..', n)

Exception: hasan is not welcomed

# try except
def div(x,y):
    try:
        result=x/y
    except ZeroDivisionError:
        result='not possible .. we can not divide by zero'
    except TypeError:
        result = int(x) / int(y)
    else:
        print('well done.. no problems')
    finally:
        print('thank you...')
    return result
```

## re
```python
txt='Hi, how are you today?'
re.search('my', txt)!=None
# False

# . = any character, except a new line 
print(re.search('a.','aa') !=None)
print(re.search('a.','ab') !=None)
print(re.search('a.','a1') !=None)
print(re.search('a.','a') !=None)
print(re.search('a.','a\n') !=None)
# True
# True
# True
# False
# False


# ? = match 0 or 1 repetitions
print(re.search('ba?b','bb') !=None)
print(re.search('ba?b','bab') !=None)
print(re.search('ba?b','abab') !=None)
print(re.search('ba?b','baab') !=None)
# True
# True
# True
# False

# # = match 0 or more repetitions
print(re.search('ba*b','bb') !=None)
print(re.search('ba*b','bab') !=None)
print(re.search('ba*b','baaaaaab') !=None)
print(re.search('ba*b','baaaaaaaaagb') !=None)
# True
# True
# True
# False

# + = match 1 or more
print(re.search('ba+b','bb') !=None)
print(re.search('ba+b','bab') !=None)
print(re.search('ba+b','baaaab') !=None)
print(re.search('ba+b','baaaaaab') !=None)
# False
# True
# True
# True

# {} = match a range of number of times
print(re.search('ba{1,3}b','bb') !=None)
print(re.search('ba{1,3}b','bab') !=None)
print(re.search('ba{1,3}b','baab') !=None)
print(re.search('ba{1,3}b','baaab') !=None)
print(re.search('ba{1,3}b','baaaab') !=None)
# False
# True
# True
# True
# False

# ^ = matches start of a string
print(re.search('^a','abc') !=None)
print(re.search('^a','abcde') !=None)
print(re.search('^a',' abc') !=None)
print(re.search('^a','bc') !=None)
# True
# True
# False
# False

# $ = matches at the end of string
print(re.search('a$','bca') !=None)
print(re.search('a$','bca ') !=None)
print(re.search('a$','bcb') !=None)
# True
# False
# False

print(re.search('^ab.ta$','aca') !=None)
print(re.search('^a.a$','aca') !=None)
print(re.search('^a.a$','a a') !=None)
print(re.search('^a.a$','abbca') !=None)
print(re.search('^a.?a$','aca') !=None)
print(re.search('^a.*a$','aabcdeca') !=None)
print(re.search('^a.{1,3}a$','aca') !=None)
# True
# True
# False
# True
# True
# True

# [] are used to specify a set of chracters to match
print(re.search('[123abc]','aca') !=None)
print(re.search('[1-5a-d]','aca') !=None)
# True
# True

# () are used to create a group of characters (together) to match
print(re.search('(abc){2,3}','abc') !=None)
print(re.search('(abc){2,3}','abcabc') !=None)
print(re.search('(abc){2,3}','abcabcabc') !=None)
print(re.search('(abc){2,3}','abcabcabcabc') !=None)
print(re.search('^(abc){1,2}$',' abcabc') !=None)
# False
# True
# True
# True
# False

# | is used as logical operator
print(re.search('abc|123','a') !=None)
print(re.search('abc|123','123') !=None)
print(re.search('abc|123','abc') !=None)
# False
# True
# True

# \ is used with special characters
print(re.search('\?','Hi, how are you today?') !=None)
# True

# special characters
# \d = matches any decimal digit [0-9]
print(re.search('\d','my name is hasan, my age is 42 and my salary is 2000') !=None)
re.findall('\S+','my name is hasan, my age is 42 and my salary is 2000') 
# True
# ['my',
#  'name',
#  'is',
#  'hasan,',
#  'my',
#  'age',
#  'is',
#  '42',
#  'and',
#  'my',
#  'salary',
#  'is',
#  '2000']

# \d = any decimal digit
# \D = any non-digit character
# |w = any alphanumeric 
# \W = any non-alphanumeric character
# \s = any white space
# \S = any non-whitespace character
# \. = the character .
# \t = tab
# \n = new line

# functions 
# re.findall()
# re.split()

```

## numpy
```python
'''Execise: how to find the person who participate most'''
f=open('../shared/datafiles/chat.txt', 'r')
lines=f.readlines()

# first way
names = [line.split()[0] for line in lines[::2]]
freq = {}
for n in names:
    if n not in freq:
        freq[n] = 1
    else:
        freq[n] += 1
freq
sorted(freq.items(), key=lambda x:x[1], reverse=True)
# [('Emily', 122),
#  ('John', 99),
#  ('Michael', 98),
#  ('William', 91),
#  ('Jack', 90),
#  ('Elizabeth', 89),
#  ('Emma', 87),
#  ('Mary', 86),
#  ('Jayden', 83),
#  ('Daniel', 77)]

# second way
import numpy as np
names = [line.split()[0] for line in lines[::2]]
np.unique(names, return_counts=True)
# This retruns:
# (array(['Daniel', 'Elizabeth', 'Emily', 'Emma', 'Jack', 'Jayden', 'John',
#         'Mary', 'Michael', 'William'], dtype='<U9'),
#  array([ 77,  89, 122,  87,  90,  83,  99,  86,  98,  91]))

# Then we do 
sorted(list(zip(np.unique(names, return_counts=True)[0],np.unique(names, return_counts=True)[1])), key=lambda r:r[1])[-1]
# ('Emily', 122)

# Third way (use python with bash shell)
f=open('names.txt','w')
for n in names:
    f.write(n + '\n')
f.close()

# shell
!cat names.txt | sort | uniq -c | sort -nr
    # 122 Emily
    #  99 John
    #  98 Michael
    #  91 William
    #  90 Jack
    #  89 Elizabeth
    #  87 Emma
    #  86 Mary
    #  83 Jayden
    #  77 Daniel

# fourth way: create a new .py file and use the bash shell in terminal (e.g: code.py)
f=open('../shared/datafiles/chat.txt', 'r')
lines=f.readlines()
names = [line.split()[0] for line in lines[::2]]
print('\n'.join(names))

# In terminal
~./code.py | sort | uniq -c | sort -nr

'''Give file as parameter'''
import sys
file_name = sys.stdin()
f = open(file_name)
```
