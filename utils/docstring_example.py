FUNCTION_DOCSTRING_TEMPLATE = """
    <One line function description here>

    <More detailed description here>

    Parameters
    ----------
    param1 : <replace by type of parameter> 
        <Detailed description>
    param2 : <replace by type of parameter>
        <Detailed description>

    Returns
    -------
    return_value : <replace by return value type> 
        <Detailed description?
    
    Raises <Include this if function raises errors>
    ------
    AttributeError
        <Some explanation goes here>
    ValueError
        <Some explanation goes here>

    Examples
    --------
    <Here's a dummy example. Recommended to have `Examples` section for important
    functions. Examples must demonstrate how to set up and call the function.>
    >>> from sklearn.preprocessing import OneHotEncoder
    >>> enc = OneHotEncoder(handle_unknown='ignore')
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)

    Notes
    -----
    <Optional section. Can add references here>
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod 
    tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, 
    quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo 
    consequat.
    """

GENERATOR_DOCSTRING_TEMPLATE = """
    <One line generator description here>

    <More detailed description here>

    Parameters
    ----------
    param1 : <replace by type of parameter> 
        <More detailed description>
    param2 : <replace by type of parameter>
        <More detailed description>

    Yields 
    ------
    int
        <Some description goes here>
    
    Raises <Include this if generator raises errors>
    ------
    AttributeError
        <Some explanation goes here>
    ValueError
        <Some explanation goes here>
    """

CLASS_DOCSTRING_TEMPLATE = """
    <Class description here>
    
    Parameters <Constructor Parameters>
    ----------
    param1 : <replace by type of parameter> 
        <The first parameter. More detailed description>
    param2 : <replace by type of parameter>
        <The second parameter. More detailed description>
    
    Attributes
    ----------
    attr1 : <replace by type of parameter> 
        <More detailed description>
    attr2 : <replace by type of parameter> 
        <More detailed description>
    """
