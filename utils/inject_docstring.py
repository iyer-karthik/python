from docstring_example import (FUNCTION_DOCSTRING_TEMPLATE, 
                              CLASS_DOCSTRING_TEMPLATE)
import ast 
import astor
try:
    from tokenize import open as fopen
except ImportError:
    fopen = open
import sys

#-------------------------------------------------------------------------------
# Helper class to modify Python AST. Must inherit from ast.NodeTransformer
#-------------------------------------------------------------------------------
class DocstringInjector(ast.NodeTransformer):
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visits a class node in an abstract syntax tree and transforms
           the node by injecting templated docstring if no docstring is 
           detected. Doesn't modify the class node if docstring is detected.

        Parameters
        ----------
        node: ast.ClassDef
            A node representing class in a Python AST

        Returns
        -------
        node: ast.ClassDef
            Modified node if no docstring is detected for the given
            class node; existing node otherwise
        """

        if not isinstance(node, ast.ClassDef):
            raise ValueError("Can only modify a class node")
        
        # First node of a class body must is of the type `ast.Expr` iff
        # class has docstring
        class_body_nodes = node.body
        _first_node, *_ = class_body_nodes
        if not isinstance(_first_node, ast.Expr):
            _expr_node = ast.Expr(value=ast.Str(s=CLASS_DOCSTRING_TEMPLATE))
            node.body.insert(0, _expr_node) # Insert docstring at the first child node
            ast.fix_missing_locations(node) # Line offset   
        return node


    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visits a function node in an abstract syntax tree and transforms
           the node by injecting templated docstring if no docstring is
           detected. Doesn't modify the function node if docstring is detected. 
           By numpy convention __init__ methods should have no docstrings.

        Parameters
        ----------
        node: ast.FunctionDef
            A node representing function in a Python AST

        Returns
        -------
        node: ast.FunctionDef
            Modified node if no docstring is detected for the given
            function node; existing node otherwise
        """

        if not isinstance(node, ast.FunctionDef):
            raise ValueError("Can only modify a function node")
        
        # First node of a function body must is of the type `ast.Expr` iff
        # function has docstring
        function_body_nodes = node.body
        _first_node, *_ = function_body_nodes
        if not isinstance(_first_node, ast.Expr):
            _expr_node = ast.Expr(value=ast.Str(s=FUNCTION_DOCSTRING_TEMPLATE))
            node.body.insert(0, _expr_node) # Insert docstring at the first child node
            ast.fix_missing_locations(node) # Line offset   
        return node
    

    def visit_Module(self, node: ast.Module):
        "TO DO - Insert module level docstring"
        pass

#-------------------------------------------------------------------------------
# Functions that modify source code

def add_templated_docstring_to_source(source_code):
    """Add templated docstring in given Python code
       
    Given Python code this function detects all functions/ methods and classes 
    in the code, detects if the each of the functions/ methods and classes have 
    docstrings and injects associated templated docstring if no docstring exists. 

    This function only potentially modifies a given piece of code if 
    functions/ methods or classes exist in the given code. Otherwise the 
    funtion doesn't modify the given code.
    
    Parameters
    ----------
    source_code : str
        string with syntactically correct Python code

    Returns
    -------
    modified_source_code: str
        modified python code with injected docstrings
    """

    my_docstring = DocstringInjector()
    inject_docstring = my_docstring.visit

    # Convert source code to an AST (Abstract Syntax Tree)
    _starting_node = ast.parse(source_code)
    _all_descendent_nodes = ast.walk(_starting_node)

    for node in _all_descendent_nodes:
        if isinstance(node, ast.FunctionDef) and node.name != '__init__':
            inject_docstring(node) # only injects templated docstring if no 
                                   # function docstring is detected. Not 
                                   # applicable to __init__ methods.
        elif isinstance(node, ast.ClassDef):
            inject_docstring(node) # only injects templated docstring if no 
                                   # class docstring is detected.
    
    # Covert the modified AST back to source code
    modified_source_code = astor.to_source(node=_starting_node)
    return modified_source_code

def inject_templated_docstring(filepath, inplace=True):

    # TO DO: add functionality for non inplace injection

    try:
        with fopen(filepath) as f:
            fstr = f.read()
    except IOError:
        if filepath != 'stdin':
            raise
        sys.stdout.write('\nReading from stdin:\n\n')
        fstr = sys.stdin.read()
    
    if not fstr.endswith('\n'):
        fstr += '\n'
    
    modified_fstr = add_templated_docstring_to_source(source_code=fstr)
    with open(filepath, 'w') as f:
        f.write(modified_fstr)
    
    print("Modified source succesfully")
#-------------------------------------------------------------------------------

# TO DO: make this into a command line utility
                           
