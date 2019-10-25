from docstring_example import (FUNCTION_DOCSTRING_TEMPLATE, 
                               GENERATOR_DOCSTRING_TEMPLATE,
                               CLASS_DOCSTRING_TEMPLATE)
from collections import namedtuple
from itertools import chain
import ast 
import astor
try:
    from tokenize import open as fopen
except ImportError:
    fopen = open
import sys



class DocstringInjector(ast.NodeTransformer):
    """
    Injects docstring in Python source code.

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

    @staticmethod
    def __get_class_name(node):
        """
        Extracts class name from a node. 

        Given a Pyton AST node representing a class, this 
        function extracts the name of the class and returns it as a 
        namedtuple

        Parameters
        ----------
        source_code : str
            String containing Python code

        Returns
        -------
        ClassDetails : namedtuple
            Each ClassDetails object has the following attributes:
            name - name of class (str)
        
        Returns None if the given node does not represent a class
        """
        ClassDetails = namedtuple("ClassDetails", ["name"])
        
        # TODO: Raise error if node is not a class def

        if isinstance(node, ast.ClassDef):
            return ClassDetails(name=node.name)

    @staticmethod
    def __get_function_name_and_arguments(node):
        """
        Extracts function name and associated arguments from a node. 
        Works for nested functions and methods too.

        Given a Pyton AST node representing a method/ function, this 
        function extracts the name and associated arguments and returns 
        them as a namedtuple
        
        Parameters
        ----------
        source_code: str
            String containing Python code

        Returns
        -------
        FunctionDetails : namedtuple
            Each FunctionDetails object has the following attributes:
            name - name of function (str)
            args - arguments for the function ({str})
        
        Returns None if the given code does not contain any function.
        """
        FunctionDetails = namedtuple("FunctionDetails", ["name", "args"])
        
        # TODO: Raise error if node is not a function def

        if isinstance(node, ast.FunctionDef):
            function_args = node.args
            arg_nodes = function_args.args
            kwarg_nodes = function_args.kwonlyargs
            args =  set(arg_node.arg for arg_node in chain(arg_nodes, kwarg_nodes))

            return FunctionDetails(name=node.name, 
                                   args=args)


    # Helper function to modify Python AST. 
    @staticmethod
    def __modify_node(node, templated_docstring: str):
        """
        Visits a node in an abstract syntax tree, detects if it is a node
        corresponding to a function or a class definition and transforms
        the node by injecting templated docstring if no docstring is 
        detected. 
        
        Does not modify the node if docstring is detected.
        TODO: Introduce an overwrite option

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
    def _add_templated_docstring_to_source(source_code: str):
        """
        Add templated docstring to given Python code
        
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

        # Only inject templated docstring if none found. 
        # Only non-private methods/ functions and classes get modified.
        for node in _all_descendent_nodes:
            if isinstance(node, ast.FunctionDef) and not(node.name.startswith('__')):
                f_name = DocstringInjector.__get_all_function_details(node).name

                parameter_template = "".join("{} : <replace by type of parameter>\n\t\t<Detailed description>\n    ".format(x)\
                                        for x in DocstringInjector.__get_all_function_details(node).args )

                # Detect if node corresponds to a generator or a normal function
                if any(isinstance(descendent_node, ast.Yield) for descendent_node in ast.walk(node)):
                    __templated_docstring = GENERATOR_DOCSTRING_TEMPLATE.format(f_name, parameter_template)
                    DocstringInjector.__modify_node(node, 
                                                    templated_docstring=__templated_docstring)
                else:
                    __templated_docstring = FUNCTION_DOCSTRING_TEMPLATE.format(f_name, parameter_template)
                    DocstringInjector.__modify_node(node,
                                                    templated_docstring=__templated_docstring)
                
            elif isinstance(node, ast.ClassDef):
                DocstringInjector.__modify_node(node,
                                                templated_docstring=CLASS_DOCSTRING_TEMPLATE)

        # Convert the modified AST back to source code. 
        # Note that `_starting_node` has now been modified.
        modified_source_code = astor.to_source(node=_starting_node)
        return modified_source_code


    @staticmethod
    def inject_templated_docstring(filepath: str):
        """
        Inject docstring inplace to a file containing Python code
        
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
            modified_fstr = DocstringInjector._add_templated_docstring_to_source(source_code=fstr)
            with open(filepath, 'w') as f:
                f.write(modified_fstr)
            print("Modified source succesfully!")
        except (ImportError, IndentationError, SyntaxError):
            raise
#------------------------------------------------------------------------------

# TO DO: make this into a command line utility
                           
# TO DO: inject docstring intelligently. Detect arguments in a function or 
# class __init__ signature and inject the templated docstring accordingly


