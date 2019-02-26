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
