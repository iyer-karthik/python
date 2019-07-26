from docstring_example import (FUNCTION_DOCSTRING_TEMPLATE, 
                               GENERATOR_DOCSTRING_TEMPLATE,
                               CLASS_DOCSTRING_TEMPLATE)
import ast 
import astor
try:
    from tokenize import open as fopen
except ImportError:
    fopen = open
import sys


class DocstringInjector(ast.NodeTransformer):
    """Injects docstring in Python source code.

    Detects functions/ methods/ generators/classes in python source code
    and injects associated templated docstrings if none exist.

    Note: Only non-private functions/ methods are modified. Modification 
    happens inplace.

    Examples
    --------
    >>> filepath = 'your_python_source_filepath'
    >>> DocstringInjector.inject_templated_docstring(filepath)
    OneHotEncoder(handle_unknown='ignore')
    """

    # Helper function to modify Python AST. 
    @staticmethod
    def _modify_node(node):
        """Visits a node in an abstract syntax tree, detects if it is a node
           corresponding to a function or a class definition and transforms
           the node by injecting templated docstring if no docstring is 
           detected. Doesn't modify the node if docstring is detected.

        Parameters
        ----------
        node: ast.ClassDef or ast.FunctionDef
            A node representing class/function/method in a Python AST

        Returns
        -------
        node: ast.ClassDef or ast.FunctionDef
            Modified node if no docstring is detected for the given
            class node; existing node otherwise
        """

        if not (isinstance(node, ast.ClassDef) or isinstance(node, ast.FunctionDef)):
            raise ValueError("Can only modify a class or a function/ method node")
        
        _first_node, *_ = node.body  # First node is of the type `ast.Expr` iff 
                                     # no docstring exists
        if isinstance(node, ast.ClassDef):
            if not isinstance(_first_node, ast.Expr):
                _expr_node = ast.Expr(value=ast.Str(s=CLASS_DOCSTRING_TEMPLATE))
                node.body.insert(0, _expr_node) # Insert docstring at the first child node
                ast.fix_missing_locations(node) # Line offset   
            return node
        
        elif isinstance(node, ast.FunctionDef):
            # Detect if the function is a generator. Inject yield template. 
            if any(isinstance(descendent_node, ast.Yield) 
                   for descendent_node in ast.walk(node)):
                if not isinstance(_first_node, ast.Expr):
                    _expr_node = ast.Expr(value=ast.Str(s=GENERATOR_DOCSTRING_TEMPLATE))
                    node.body.insert(0, _expr_node) # Insert docstring at the first child node
                    ast.fix_missing_locations(node) # Line offset   

            # Otherwise inject usual function docstring template
            else:
                if not isinstance(_first_node, ast.Expr):
                    _expr_node = ast.Expr(value=ast.Str(s=FUNCTION_DOCSTRING_TEMPLATE))
                    node.body.insert(0, _expr_node) # Insert docstring at the first child node
                    ast.fix_missing_locations(node) # Line offset   
            return node
    
        return node
#-------------------------------------------------------------------------------
# Functions that modify source code
    @staticmethod
    def add_templated_docstring_to_source(source_code):
        """Add templated docstring to given Python code
        
        Given Python code this function detects all functions/ methods and 
        classes in the code, checks if the each of the functions/ methods and 
        classes have docstrings and injects associated templated docstring if no 
        docstring exists. This injection is done only for non-private 
        functions/methods.

        This function only potentially modifies a given piece of code if 
        functions/ methods or classes exist in the given code. Otherwise the 
        funtion doesn't modify the code.
        
        Parameters
        ----------
        source_code : str
            string with syntactically correct Python code

        Returns
        -------
        modified_source_code: str
            modified python code with injected docstrings
        """
        # Convert source code to an AST (Abstract Syntax Tree)
        _starting_node = ast.parse(source_code)
        _all_descendent_nodes = ast.walk(_starting_node)

        # Only injects templated docstring if none found. Only non-private 
        # methods/ functions and classes get injected. 
        for node in _all_descendent_nodes:
            modify_node_bool = (isinstance(node, ast.FunctionDef) and\
                                not(node.name.startswith('__')) or\
                                isinstance(node, ast.ClassDef))
            
            if modify_node_bool:
                DocstringInjector._modify_node(node)

        # Covert the modified AST back to source code
        modified_source_code = astor.to_source(node=_starting_node)
        return modified_source_code

    @staticmethod
    def inject_templated_docstring(filepath: str):

        """Inject docstring inplace to a file containing Python code
        
        Given a filepath that contains Python code function detects all 
        functions/ methods and classes in the code, checks if the each of the 
        functions/ methods and classes have docstrings and injects associated 
        templated docstring if no docstring exists. This injection is 
        done only for non-private functions/methods. Finally this function
        writes to the same location as provided in the argument i.e modifies
        source code in place. 

        Parameters
        ----------
        filepath : str
            absolute or relative filepath for file containing Python code

        Returns
        -------
        None
        
        Raises
        ------
        ImportError, SyntaxError, IndentationError
            If Python source code is syntactically incorrect
        """
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
        
        try:
            modified_fstr = DocstringInjector.add_templated_docstring_to_source(source_code=fstr)
            with open(filepath, 'w') as f:
                f.write(modified_fstr)
            print("Modified source succesfully")
        except (ImportError, IndentationError, SyntaxError):
            raise
#-------------------------------------------------------------------------------

# TO DO: make this into a command line utility
                           
