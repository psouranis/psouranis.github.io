---
layout: post
title: Python metaclasses
date: 2025-06-11 12:40:16
description: Python metaclasses
tags: Python Metaclasses
categories: Python
thumbnail: assets/post_images/numpy/numpy_go_brr.png
giscus_comments: true
toc: 
  beginning: true
---


This post serves as a reference from a great talk from PyCon US ([Link](https://www.youtube.com/watch?v=fEm0vi8SpkI)) where Jason C. McDonald discusses Python's metaclasses.
The goal of this post is to summarize what was said in that video and put it in a readable format.

## Python Classes

<br>
<div style="text-align: center;">
  <img src="/assets/post_images/metaclasses/blueprint.png" style="width: 50%; height: auto;">
</div>
<br>

Alright, let's first start with something that we are all familiar. Python Classes. Everyone used them at some time and everyone is familiar with them.
Python classes are like blueprints. You can think of a blueprint of a house that is built where you can then use it to build multiple houses and each house 
will look like the blueprint and if you change the blueprint then each house you create from now on will be different but if you change one of the houses
you are only changing that one house and it's not affecting the others.

In Python, we say everything is an `object`. And so classes are objects too. Going back to our analogy of the blueprint, if you try to apply that thinking to the blueprint 
itself you will realize that blueprints are based on templates. You can have a template for example that creates blueprints of houses where the specs are on meters or a template
that you create blueprints of houses in feet.

So those classes (blueprints) are still objects and they are instances of the `metaclass` which is the template that I mentioned above. So we can have multiple templates
but the one built-in we use everytime in Python is called `type`. 

We have used `type` multiple times but for different reason. Like for example when we want to check the type of an instance

```python
x = np.array([1,2,3,4])

print(type(x))
# <class 'numpy.ndarray'>
```

You notice that the type is `class`!. So `type` is actually Python's built-in default metaclass. And the thing is that you can actually create a class dynamically using `type`

```python
type("Foo", (), {})
__main__.Foo
```

> The `class` statement isn't just syntatictic sugar, it does some extra things, like setting a `__qualname__` and `__doc__` properties.


The funny thing about `type` is that is of type `type` . 

1. Name of the classes.
2. Bases (Inheritance).
3. Keyword arguments (such as `metaclass`)

You can see more about dynamic creation of a class at [datamodel](https://docs.python.org/3/reference/datamodel.html). 

Alright, now that we have a graps of what metaclasses are, let's see what the use cases for using one are since this is what we are interested in.

## Metaclasses

What does a metaclass do? There are 4 things a metaclass manages in Python.

<br>
<div style="text-align: center;">
  <img src="/assets/post_images/metaclasses/metaclass_do.png" style="width: 50%; height: auto;">
</div>
<br>

1. The first is class instantiation.


```python
__call__()
__new__()
```

Metaclass calls `__call__` which in turn calls `__new__` from `Class` (and if it returned an instance of `Class` it will also call `Class.__init__`)

<br>
<div style="text-align: center;">
  <img src="/assets/post_images/metaclasses/instance_creation.png" style="width: 50%; height: auto;">
</div>
<br>


2. Class Attribute Storage.


```python
@classmethod
__prepare__()
```

This method is called before the class body is executed and it must return a dictionary-like object that's used as the local namespace for all the code from the class body. It was added in Python 3.0, see PEP-3115.

If your `__prepare__` returns an object `x` then this:

```python
class Class(metaclass=type):
    a = 1
    b = 2
    c = 3
```

Will make the following changes to x:

```python
x['a'] = 1
x['b'] = 2
x['c'] = 3
```


3. Method Resolution Order.

```python
mro()
```

The `type` metaclass defines the `type.mro()` which sets the method resolution order for its instances. It is called at class instantiation, and its result is stored in `__mro__` (This attribute is a tuple of classes that are considered when looking for base classes during method resolution.).


4. Inheritance.

```python
__isinstancecheck__()
__subclasscheck__()
```

`__isinstancecheck__` and `__subclasscheck__` are powerful metaclass methods that allow you to customize 
how the built-in isinstance() and issubclass() functions behave for classes defined with that metaclass. They provide a way to inject custom logic for determining object and class relationships, going beyond the standard inheritance model.

## Use cases 

Next and most importantly, we will explore some use cases that the metaclasses make sense to use.

### Use case no. 1 - Singletons

Singletons allow us to make sure that we instantiate a class only once. This could be useful
for cases that we want a global shared instance for example.

Let's see how we can create a `Singleton` metaclass.

```python
class Singleton(type)

  _instance = None

  def __call__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super().__call__(*args, **kwargs)
    return cls._instance
```

To make a metaclass we inherit from `type`. Important to remember: An `instance` method on a metaclass
is a `classmethod` on a class. So this `__call__` method will be called when we create **and** instantiate a class that inherits from `Singleton` metaclass.

```python
class Once(metaclass=Singleton)

  def __init__(self, name):
    self.name = name

once = Once("Bob")
twice = Once("Jason")
print(once.name) # Bob
print(twice.name) # Bob
```

### Use case no. 2 - Controlling Class Methods

You could...

1. Add decorators to all class methods.
2. Validate class methods.
3. Prevent some class methods.

In other words, this is as close as Python gets to compiler time errors. Let's say for example that we have
a metaclass called `Unaddable`.

```python
class Unaddable(type):

  def __new__(meta, cls, bases, attributes):
    if '__add__' in attributes:
      raise TypeError("Can't override __add__")

    return super().__new__(meta, cls, bases, attributes)
```

We don't want to fail when we instantiate an object or when we try to call `__add__`. We literally don't want any classes
using this metaclass that have `__add__` defined at all.

```python
class Misbehaving(metaclass=Unaddable):

  def __init__(self, value):
    self.value = value

  def __add__(self, other):
    return Misbehaving(self.value + self.other)
```

As soon as I run this code (without instantiating anything) we get a `# TypeError: Can't override __add__` as soon as I run the code.
This could be used in cases that is very crucial that I should not define something I'm not suppose to.

### Use case no. 3 - Changing Class Namespace

In Python, we typically store everything in a class in a dictionary, which is all well and good, but sometimes you want to store it a little differently.

```python
class EnumType(type):

  @classmethod
  def __prepare__(metacls, cls, bases, **kwds):
    metacls._check_for_existing_members_(cls, bases)
    enum_dict = EnumDict(cls)
    member_type, first_enum = metacls._get_mixins_(cls, bases)

    if first_enum is not None:
      enum_dict['_generate_next_value_'] = getattr(first_enum, '_generate_next_value_', None)
    return enum_dict
```

That's how `Enum` works. It overrides the dictionary that stores those class attributes. Cause if you think about how `Enum` works, it's just a class with a bunch of class attributes. No member functions or anything. And that's because we are using the `__prepare__` method in the metaclass to change how those class attributes are being stored.

### Use Case no. 4 - Manipulating MRO

```python
class Arcana

  def __repr__(self):
    return "Arcana"

  def magic_word(self):
    print("Abracadabra")
```

Let's say that I want my class to look at the following class as an absolute last resort. I don't want to override anything, not even an object. But if it's not anywhere at all, I want to check at the code above last.

One way to do that, is to define a metaclass and change the `mro`.

```python
class KnowsArcana(type):
  def mro(cls):
    return [*super().mro(), Arcana]

class MyClass(metaclass=KnowsArcana):
  pass

my_obj = MyClass()
my_obj.magic_word() # "Abracadabra!"
print(repr(my_obj)) # "<__main__.MyClass object at ...>"
```

You see that if I try to call `magic_word()` it finds it but if I try to use `print` it will use the default `__repr__` of `type`.

### Use Case no. 5 - Class Dunder Methods

```python
class AddressBook:
  mutable_dict = dict()

  @classmethod
  def __getattr__(cls, name):
    try:
      return cls.mutable_dict[name]
    except KeyError:
      return super().__getattribute__(name)
    
  @classmethod
  def store(cls, entries):
    for key, value in entries.items():
      key = key.replace(" ", "_")
      cls.mutable_dict[key] = value 

AddressBook.store({"Bob Smith": "bob@example.com"})
print(AddressBook.Bob_Smith)

# AttributeError: type object 'AddressBook' has no attribute 'Bob_Smith'
```

If you want to override dunder methods of the class you need to do it on the metaclass.

```python
class DottedDict(type):

  @classmethod
  def __prepare__(metacls, name, bases):
    return {
      'mutable_dict': dict()
    }

  def __getattr__(self, name):
    try:
      return self.mutable_dict[name]
    except:
      return super().__getattribute__(name)

  def store(cls, entries):
    for key, value in entries.items():
      key = key.replace(" ", "_")
      cls.mutable_dict[key] = value

class AddressBook(metaclass=DottedDict):
  pass

AddressBook.store(
  {
    "Bob Smith": "bob@example.com",
    "Fred Wright": "fred@example.com"
  }
)

print(AddressBook.Bob_Smith) # "bob@example.com"

```

### One More Example - Abstract Base Class

Everytime you use an abstract base class, you are using a metaclass. This is the python's source code for `ABCMeta`

```python
class ABCMeta(type):

  def __new__(mcls, name, bases, namespaces, /, **kwargs):
    cls = super().__new__(mcls, name, bases, namespace, **kwargs)
    _abc_init(cls)
    return cls
  
  def register(cls, subclass):
    return _abc_register(cls, subclass)

  def __instancecheck__(cls, instance):
    return _abc_instancecheck(cls, instance)

  def __subclasscheck__(cls, subclass):
    return _abc_subclasscheck(cls, subclass)

  def _abc_registry_clear(cls):
    _reset_registry(cls)

  def _abc_caches_clear(cls):
    _reset_caches(cls)
```

The most crucial parts of this metaclass is the `__instancecheck__` and `__subclasscheck__`.


## References

- [Metaclasses Demystified - Jason C. McDonald](https://www.youtube.com/watch?v=fEm0vi8SpkI)
- [Understanding Python's Metaclasses](https://blog.ionelmc.ro/2015/02/09/understanding-python-metaclasses/)



