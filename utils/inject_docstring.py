from docstring_example import function_docstring_example
import ast 
import astor


class MyDocstring(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        """Visits a function node in an abstract syntax tree and transforms
           the node by injecting templated docstring if no docstring is detected.
           Doesn't modify the function node if docstring is detected.

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
            _expr_node = ast.Expr(value=ast.Str(s=function_docstring_example))
            node.body.insert(0, _expr_node) # Insert docstring at the first child node
            ast.fix_missing_locations(node) # Line offset   
        return node

def inject_function_docstring(source_code):
    """Inject templated docstring for all functions in given Python code
       
    Given Python code this function detects all functions/ methods in the 
    code, detects if the each of the functions have docstrings and injects
    templated docstring if no docstring exists. 

    This function only potentially modifies a given piece of code if 
    functions/ methods exist in the given code. Otherwise the funtion doesn't
    modify the given code.
    
    Parameters
    ----------
    source_code : str
        string with syntactically correct Python code

    Returns
    -------
    modified_source_code: str
        modified python code with injected docstrings
    """

    my_docstring = MyDocstring()
    inject_docstring = my_docstring.visit

    # Convert source code to an AST (Abstract Syntax Tree)
    _starting_node = ast.parse(source_code)
    _all_descendent_nodes = ast.walk(_starting_node)

    for node in _all_descendent_nodes:
        if isinstance(node, ast.FunctionDef):
            inject_docstring(node) # only injects templated docstring if no 
                                   # docstring is detected
    
    # Covert the modified AST back to source code
    modified_source_code = astor.to_source(node=_starting_node)
    return modified_source_code
                           