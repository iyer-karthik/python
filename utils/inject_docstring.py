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

    Uses Python AST (Abstract Syntax Tree) to detect functions/ methods/ 
    generators/classes in Python source code and modify source code by 
    injecting associated templated docstrings if none detected.

    Note: Only non-private functions/ methods are modified. Modification 
    happens inplace.

    Examples
    --------
    >>> filepath = 'your_python_source_filepath'
    >>> DocstringInjector.inject_templated_docstring(filepath)
    """

    # Helper function to modify Python AST. 
    @staticmethod
    def _modify_node(node, templated_docstring: str):
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

        if ast.get_docstring(node) is None:
            _expr_node = ast.Expr(value=ast.Str(s=templated_docstring))
            node.body.insert(0, _expr_node) # Insert docstring at the 
                                            # first child node
            ast.fix_missing_locations(node) # Line offset  
        return node

#------------------------------------------------------------------------------
# Functions that modify source code
    @staticmethod
    def add_templated_docstring_to_source(source_code: str):
        """Add templated docstring to given Python code
        
        Given Python code this function detects all functions/ methods and 
        classes in the code, checks if the each of the functions/ methods and 
        classes have docstrings and injects associated templated docstring if 
        no docstring exists. This injection is done only for non-private 
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

        # Only inject templated docstring if none found. Only non-private 
        # methods/ functions and classes get modified.
        for node in _all_descendent_nodes:
            if isinstance(node, ast.FunctionDef) and not(node.name.startswith('__')):
                # Detect if node corresponds to a generator or a normal function
                if any(isinstance(descendent_node, ast.Yield)\
                       for descendent_node in ast.walk(node)):
                    DocstringInjector._modify_node(node, 
                        templated_docstring=GENERATOR_DOCSTRING_TEMPLATE)
                DocstringInjector._modify_node(node,
                    templated_docstring=FUNCTION_DOCSTRING_TEMPLATE)
                
            elif isinstance(node, ast.ClassDef):
                DocstringInjector._modify_node(node,
                    templated_docstring=CLASS_DOCSTRING_TEMPLATE)

        # Convert the modified AST back to source code. Note that `_starting_node`
        # has now been modified.
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
            print("Modified source succesfully!")
        except (ImportError, IndentationError, SyntaxError):
            raise
#------------------------------------------------------------------------------

# TO DO: make this into a command line utility
                           
# TO DO: inject docstring intelligently. Detect arguments in a function or 
# class __init__ signature and inject the templated docstring accordingly


