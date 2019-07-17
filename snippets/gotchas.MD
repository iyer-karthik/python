Learn to use 

1. `partial()` from functools

2. `namedtuple` from collections

3. `chain` from itertools (for combining multiple lists)

```python
def chain(*args):
    """Chain together several args"""
    for arg in args:
        yield from arg
        
people = ["Zeus", "Athena", "Apollo", "Poseidon"]
symbols = ["Thunderbolt", "Wisdom", "Sun", "Sea"]
countries = ["Greece", "India"]
result = chain(people, symbols, countries)
print(list(result))
```

4. `from sys import getsizeof # Memory check size for an object`

5. Items that evaluate to `False` in Python.
Anything defined as `None` or `False`.
Zero of any numeric type: 0, 0.0, 0j, Decimal(0), Fraction(0, 1)
Empty sequences and collections: `'', (), [], {}, set(), range(0), len(0)`

Everything else evaluates to `True`

`True` and `False` also can be used in an integer context, `True = 1, False = 0`

6. Break, Continue, Pass logic

Break : leave current loop, do not continue iterating

Continue : leave current loop, continue iterating

Pass : Do nothing, commonly used as a place holder for empty functions

7. "==" is used for value comparison, "is" is used for identity comparison

8. Implement type checking. Copied from https://stackoverflow.com/questions/3038033/what-are-good-uses-for-python3s-function-annotations
```python
def validate(func, locals):
    for var, test in func.__annotations__.items():
        value = locals[var]
        msg = 'Var: {0}\tValue: {1}\tTest: {2.__name__}'.format(var, value, test)
        assert test(value), msg


def is_int(x):
    return isinstance(x, int)

def between(lo, hi):
    def _between(x):
            return lo <= x <= hi
    return _between

def f(x: between(3, 10), y: is_int):
    validate(f, locals())
    print(x, y)


>>> f(0, 31.1)
Traceback (most recent call last):
   ... 
AssertionError: Var: y  Value: 31.1 Test: is_int
```
Learn to use 

1. `partial()` from functools

2. `namedtuple` from collections

3. `chain` from itertools (for combining multiple lists)

```python
def chain(*args):
    """Chain together several args"""
    for arg in args:
        yield from arg
        
people = ["Zeus", "Athena", "Apollo", "Poseidon"]
symbols = ["Thunderbolt", "Wisdom", "Sun", "Sea"]
countries = ["Greece", "India"]
result = chain(people, symbols, countries)
print(list(result))
```

4. `from sys import getsizeof # Memory check size for an object`

5. Items that evaluate to `False` in Python.
Anything defined as `None` or `False`.
Zero of any numeric type: 0, 0.0, 0j, Decimal(0), Fraction(0, 1)
Empty sequences and collections: `'', (), [], {}, set(), range(0), len(0)`

Everything else evaluates to `True`

`True` and `False` also can be used in an integer context, `True = 1, False = 0`

6. Break, Continue, Pass logic

Break : leave current loop, do not continue iterating

Continue : leave current loop, continue iterating

Pass : Do nothing, commonly used as a place holder for empty functions

7. "==" is used for value comparison, "is" is used for identity comparison

8. Implement type checking 
```python
def validate(func, locals):
    for var, test in func.__annotations__.items():
        value = locals[var]
        msg = 'Var: {0}\tValue: {1}\tTest: {2.__name__}'.format(var, value, test)
        assert test(value), msg


def is_int(x):
    return isinstance(x, int)

def between(lo, hi):
    def _between(x):
            return lo <= x <= hi
    return _between

def f(x: between(3, 10), y: is_int):
    validate(f, locals())
    print(x, y)


>>> f(0, 31.1)
Traceback (most recent call last):
   ... 
AssertionError: Var: y  Value: 31.1 Test: is_int
```

9. Function Parameters as References (Fluent Python Pg 235)

The only mode of parameter passing in Python is call by sharing. That is the same
mode used in most OO languages, including Ruby, SmallTalk, and Java (this applies
to Java reference types; primitive types use call by value). Call by sharing means that
each formal parameter of the function gets a copy of each reference in the arguments.
In other words, the parameters inside the function become aliases of the actual argu‐
ments.

The result of this scheme is that a function may change any mutable object passed as
a parameter, but it cannot change the identity of those objects (i.e., it cannot alto‐
gether replace an object with another). 

```python
>>> def f(a, b):
... a += b
... return a
...
>>> x = 1
>>> y = 2
>>> f(x, y)
3
>>> x, y
(1, 2)
>>> a = [1, 2]
>>> b = [3, 4]
>>> f(a, b)
[1, 2, 3, 4]
>>> a, b
([1, 2, 3, 4], [3, 4])
>>> t = (10, 20)
>>> u = (30, 40)
>>> f(t, u)
(10, 20, 30, 40)
>>> t, u
((10, 20), (30, 40))
```

How do we guard against this? Don't change the state!
Have a copy inside the function

```python
>>> def f(a, b):
... from copy import copy
... a1 = copy(a)
... a1 += b
... return a1
...
>>> x = 1
>>> y = 2
>>> f(x, y)
3
>>> x, y
(1, 2)
>>> a = [1, 2]
>>> b = [3, 4]
>>> f(a, b)
[1, 2, 3, 4]
>>> a, b
([1, 2, [3, 4])
>>> t = (10, 20)
>>> u = (30, 40)
>>> f(t, u)
(10, 20, 30, 40)
>>> t, u
((10, 20), (30, 40))

```
