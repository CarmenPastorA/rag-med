

""" dunder variables """

def inject_dunder(module_name):
    """Injects dunder variables into the specified module"""
    import sys
    module = sys.modules[module_name]
    
    # variables
    module.__author__ = "Jesús Santamaría"
    module.__credits__ = [
        "Jesús Santamaría", 
        "Carmen Pastor"
    ]
    module.__copyright__ = "Copyright 2025, ARQA Project"
    module.__license__ = "GPL"
    module.__version__ = "0.1"
    module.__maintainer__ = "Jesús Santamaría"
    module.__email__ = "jsantamariam_externo@aemps.es"
    module.__status__ = "Development"
