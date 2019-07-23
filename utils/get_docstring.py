"""Utility script to extract class level information (name, docstring) and 
function/method levelinformation (name, arguments and docstring) from a given piece 
of Python code text. 
"""

#from functools import reduce
import ast
from itertools import chain
from collections import namedtuple


# def _get_all_class_level_expressions_recursive(class_definition_node_body):
#     """
#     Helper function to recursively that gets all class level expressions by 
#     traversing all child nodes of a ClassDef node of an AST

#     Parameters
#     -----------
#     class_definition_body: list of class_definition_Body objects?
#     """
#     if class_definition_node_body:
#         if isinstance(class_definition_node_body[0], ast.Expr):
#             yield class_definition_node_body[0]
#         yield from _get_all_class_level_expressions_recursive(
#             class_definition_node_body[1:])

#         if isinstance(class_definition_node_body[0], ast.ClassDef):
#             yield from _get_all_class_level_expressions_recursive(
#                 class_definition_node_body[0].body)


def get_all_class_name_and_docstrings(code_text):
    """Extract all class names and associated class level
    docstrings from given code. Works for nested classes too.

    Given a text of syntactically correct Python code this utility function 
    extracts names of all classes and associated class level docstrings. This 
    function returns a list of namedtuples each element of which contains class 
    level meta-information for every class deteced. 

    Parameters
    ----------
    
    code_text: str
        String containing Python code

    Returns
    -------
    [ClassDetails]: List of namedtuples
        A list of namedtuple called ClassDetails. Each ClassDetails
        object has the following attributes:
        name - name of class (str)
        docstring - docstring of class (str)
    
    Returns an empty list if the given code does not contain any class.
    """
    ClassDetails = namedtuple("ClassDetails", ["name", "docstring"])
    
    # Parse the code text to get the starting node of the AST
    starting_node = ast.parse(code_text)
    
    # Recursively yield all descendant nodes in the tree starting at `starting_node`
    # Used and exhausted exactly once afterwards, so okay to load lazily. 
    _all_descendent_nodes = ast.walk(starting_node)
    
    return [ClassDetails(name=node.name, docstring=ast.get_docstring(node)) 
            for node in _all_descendent_nodes
            if isinstance(node, ast.ClassDef)]


def get_all_function_details(code_text: str):
    """Extract all function names, associated arguments and associated 
    docstrings from given code. Works for nested functions and methods too.

    Given a text of syntactically correct Python code this utlitiy function 
    extracts names of all functions and methods, their respective arguments and 
    docstrings. This function returns a list of namedtuples each element of 
    which contains functional meta-information for every function/ method 
    detected.  

    Parameters
    ----------
    
    code_text: str
        String containing Python code

    Returns
    -------
    [FunctionDetails]: List of namedtuples
        A list of namedtuple called FunctionDetails. Each FunctionDetails
        object has the following attributes:
        name - name of function/ method (str)
        args - set of all arguments to the function/ method (Set(Str))
        docstring - docstring of function/ method (str)
    
    Returns an empty list if the given code does not contain any methods/
    functions.
    """
    FunctionDetails = namedtuple("FunctionDetails", ["name", "args", "docstring"])
    
    # Parse the code text to get the starting node of the AST
    starting_node = ast.parse(code_text)
    
    # Recursively yield all descendant nodes in the tree starting at `starting_node`
    # Used and exhausted exactly once afterwards, so okay to load lazily. 
    _all_descendent_nodes = ast.walk(starting_node)
    
    return [FunctionDetails(name=node.name, 
                            args=_get_function_args(node), 
                            docstring=ast.get_docstring(node)) 
            for node in _all_descendent_nodes
            if isinstance(node, ast.FunctionDef)]


def _get_function_args(function_definition_node):
    """Helper function to extract all arguments (keyworded and otherwise) from
    a FunctionDef node of an AST
    
    Parameters
    ----------
    function_definition_node : [type]
        [description]
    """
    function_arg_node = function_definition_node.args
    arg_nodes = function_arg_node.args
    kwonly_arg_nodes = function_arg_node.kwonlyargs
    return set(arg_node.arg for arg_node in chain(arg_nodes, kwonly_arg_nodes))
