

"""
Utilities
"""

import os
import sys
# Add parent directory to path to access shared modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from shared import dunder_info
dunder_info.inject_dunder(__name__) # injects the variables

def normalize_doc_id(doc_id: str) -> str:
    return doc_id.replace("FT_", "").replace("_", " ").strip()

def get_text(input_file):
    with open(input_file, "r", encoding="utf8") as document:
        text = document.read()
    return text

def get_lines(input_file, encoding="utf8"):
    with open(input_file, "r", encoding=encoding) as document:
        lines = document.read().strip().split('\n')
    
    return lines

def vprint(string, verbose):
    if (verbose):
        print(string)

def get_dict_from_json(filename):
    import json
    with open(filename, encoding='utf8') as f_in:
        return(json.load(f_in))

# Turn a Unicode string to plain ASCII
def unicodeToAscii(s, all_letters):
    import unicodedata
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def clean(string='', punct='', sw=None):
    string1 = ''.join([word for word in string if word not in punct])
    string2 = string1.lower()
    string_cleaned = ' '.join([word for word in string2.split() if word not in sw])
    
    return string_cleaned

def get_pickle(path):
    import pickle as pic
    with open(path, "rb") as fp:
        obj = pic.load(fp)
    return obj

def save_pickle(path, obj):
    import pickle as pic
    with open(path, "wb") as fp:
        pic.dump(obj, fp)

def save_dict_to_json(file_dict, dictionary, indent=2):
    import json
    d = {}
    for k in dictionary:
        v = dictionary.get(k)
        if isinstance(v, set):
            d[k] = list(v)
        else:
            d[k] = v
    with open(file_dict, 'w', encoding='utf8') as fp:
        json.dump(d, fp, ensure_ascii=False, indent=indent)

class StemProcessor:
    """
    Utility class for performing iterative stemming on tokens, strings, and lists of strings.

    A single stemmer instance (e.g., PorterStemmer) is passed to the constructor
    and reused across all stemming operations.

    Example:
        >>> from nltk.stem import PorterStemmer
        >>> stemmer = PorterStemmer()
        >>> sp = StemProcessor(stemmer)
        >>> sp.stem_token("running")
        'run'
        >>> sp.stem_line("Running faster and barking loudly")
        'run faster and bark loudli'
        >>> sp.stem_list(["Dogs running", "Easily done"])
        ['dog run', 'easili done']
    """

    def __init__(self, stemmer):
        """
        Initializes the StemProcessor with a stemmer.

        Args:
            stemmer: A stemming object, such as nltk.stem.PorterStemmer.
        """
        self.stemmer = stemmer

    def _stem_token(self, token):
        """
        Iteratively applies stemming to a token until the result no longer changes.

        Args:
            token (str): A single word to be stemmed.

        Returns:
            str: The final stable stem of the input token.
        """
        stem = self.stemmer.stem(token.lower())
        stem2 = stem
        while True:
            stem = self.stemmer.stem(stem2)
            if stem != stem2:
                stem2 = stem
            else:
                break
        return stem

    def stem_line(self, line):
        """
        Applies stemming to each token in a string.

        Args:
            line (str): A line of text to be stemmed.

        Returns:
            str: The input string with each token stemmed.
        """
        tokens = line.split()
        return ' '.join([self._stem_token(token) for token in tokens])

    def stem_list(self, lines):
        """
        Applies stemming to each string in a list of strings.

        Args:
            lines (list of str): A list of text lines to be stemmed.

        Returns:
            list of str: A list of stemmed strings.
        """
        return [self.stem_line(line) for line in lines]


def check_equal(lst):
    """
    Checks if all elements in a list are equal.
    Method "One-liner"
    
    Example:
        >>> check_equal(['a', 'a', 'b'])
        False
    """
    eq = lst[1:] == lst[:-1]
    return eq

def path_leaf(path):
    """
    Extracts the final component (filename or directory name) from a given file path,
    regardless of the operating system or path format.

    This function handles both Windows and Unix-style paths, and works even if the path
    ends with a trailing slash or backslash.

    Args:
        path (str): A file or directory path. Example: '\\\\a\\\\b\\\\c\\\\'

    Returns:
        str: The last part of the path. Example: 'c'

    Example:
        >>> path_leaf("/home/user/file.txt")
        'file.txt'
        >>> path_leaf("C:\\\\Users\\\\Admin\\\\Documents\\\\")
        'Documents'
    """
    import ntpath
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)



def get_file_filename_ext(path):
    """
    Extract filename with extension, filename without extension, and extension from paths.
    No matter what the operating system or path format could be.
    If the file ends with a slash, all 3 will be empty
    Parameters
    ----------
    path : str
        >>> '\\a\\b\\c.txt'
    Returns
    -------
    filename : str
        >>> 'c.txt'
    name : str
        >>> 'c'
    ext : str
        >>> '.txt'
    """
    import os
    import ntpath
    filename = ntpath.basename(path)
    (name, ext) = os.path.splitext(filename)
    return filename, name, ext





def deduplicate_ordered(seq, remove=True, string_for_equal_elements=''):
    """
    Removes duplicate elements from a sequence while preserving the original order.
    This is the fastest one method:
    Python is a dynamic language, and resolving seen.add each iteration is more costly than resolving a local variable. 
    seen.add could have changed between iterations, and the runtime isn't smart enough to rule that out. 
    To play it safe, it has to check the object each time.
    
    If `remove` is True (default), duplicate elements are omitted entirely.
    If `remove` is False, duplicates are replaced by `string_for_equal_elements`.
    
    Args:
        seq (list): Input sequence containing potentially repeated elements.
        remove (bool): Whether to remove duplicates (True) or replace them (False).
        string_for_equal_elements (str): Replacement string for duplicates if `remove` is False.
    
    Returns:
        list: A list with duplicates removed or replaced, preserving the order of first appearances.
    
    Example:
        >>> deduplicate_ordered(["a", "b", "a", "c", "b"])
        ['a', 'b', 'c']

        >>> deduplicate_ordered(["a", "b", "a", "c", "b"], remove=False, string_for_equal_elements="_")
        ['a', 'b', '_', 'c', '_']
    """
    seen = set()
    seen_add = seen.add
    if remove:
        return [x for x in seq if not (x in seen or seen_add(x))]
    return [x if not (x in seen or seen_add(x)) else string_for_equal_elements for x in seq]





def trackcalls(func):
    """
    Decorator to set a function attribute, to test if the function has been called.
    To use it, put it on the functions you want to check it, like this:
    
    @trackcalls
    def fn():
        pass # do stuff
    
    fn.has_been_called # check if fn has been called
    """
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.has_been_called = True
        return func(*args, **kwargs)
    wrapper.has_been_called = False
    return wrapper











def memoize(MEMO):
    """
    Decorator that remember the input and output of a function but keep the function’s behavior as-is.
    Emulate @functools.lru_cache()
    """
    def caller_fn(fn):
        def _deco(*args, **kwargs):
            import pickle
            import hashlib
            # pickle the function arguments and obtain hash as the store keys
            key = (fn.__name__, hashlib.md5(pickle.dumps((args, kwargs), 4)).hexdigest())
            # check if the key exists
            if key in MEMO:
                ret = pickle.loads(MEMO[key])
            else:
                ret = fn(*args, **kwargs)
                MEMO[key] = pickle.dumps(ret)
            return ret
        return _deco
    return caller_fn





def timefn(fn):
    """
    Decorator that prints the time it takes for a function to execute.
    """
    from functools import wraps
    import time
    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print(f"@timefn: {fn.__name__} took {t2 - t1} seconds")
        return result
    return measure_time




def search_complete_line(txt, pt, char_to_search='\n'):
    """
    searches for a given character backwards and forwards, starting from a point
    """
    # backwards
    idx_left = pt
    for idx_left in range(pt, -1, -1):
        if txt[idx_left] == char_to_search: # found
            break
    # forwards
    idx_right = skip_consecutive_characters(txt, pt)
    for idx_right in range(idx_right, len(txt)):
        if txt[idx_right] == char_to_search: # found
            break
    #
    line = txt[idx_left:idx_right]
    return line.strip()



def dump_args(func):
    """
    Decorator to print function call details.
    This includes parameters names and effective values.
    """
    import inspect
    def wrapper(*args, **kwargs):
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = ", ".join(map("{0[0]} = {0[1]!r}".format, func_args.items()))
        print(f"{func.__module__}.{func.__qualname__} ( {func_args_str} )")
        return func(*args, **kwargs)
    
    return wrapper



def stream_docs(path):
    """
    generator that yield a line from file
    :param path: path file
    :return: line of file
    """
    with open(path, 'r', encoding='utf-8') as fp:
        for line in fp:
            yield line.strip()

def get_minibatch(doc_stream, size):
    """
    returns a list with specific size
    :param doc_stream: generator of lines
    :return: list of lines
    """
    docs = []
    try:
        for _ in range(size):
            text = next(doc_stream)
            docs.append(text)
    except StopIteration:
        return None
    return docs


def repr_obj(obj):
    """
    Generate a string representation of an object based on its attributes.
    """
    cls_name = obj.__class__.__name__
    attributes = [f"{key}={value!r}" for key, value in obj.__dict__.items()]
    return f"{cls_name}({', '.join(attributes)})"



def extract_and_check_response(func):
    """
    Decorator to extract the content of the response from a model's output and check for any errors or content filtering.

    This decorator ensures that the function always returns a string. If there is an issue with the response,
    such as an error in retrieving it or if the response was filtered for inappropriate content, 
    the corresponding error message is returned as the response.

    Parameters:
    func (function): The function to be decorated, expected to return the model's response object.

    Returns:
    str: The content of the model's response, or an error message if something went wrong.
    """
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            answer = func(*args, **kwargs)
            
            # Check if the response is valid and contains choices
            if not answer or not answer.choices:
                return "Error: Did not receive a valid response from the model."

            # Check if the response was filtered
            filters = answer.choices[0].content_filter_results
            for category, result in filters.items():
                if result['filtered']:
                    return f"Warning: The response was filtered due to {category} content with severity {result['severity']}."
            
            # Extract the content of the message
            response_content = answer.choices[0].message.content
            # Return the actual response content
            return response_content

        except Exception as e:
            # In case of any exception, return the exception message as the response
            return f"Exception occurred: {str(e)}"

    return wrapper


def check_if_running(process_name):
    """
    Check if the specific 'process_name' is running
    process_name -> str
    """
    import psutil
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running


def convert_timestamp_to_date(timestamp_ms):
    """
    Converts a timestamp in milliseconds to a date string in Spanish.
    If `timestamp_ms` is -1, returns empty string.
    If locale fails, use manual conversion.
    
    Example:
        >>> convert_timestamp_to_date(1542754800000)
        '21 de noviembre de 2018'
        >>> convert_timestamp_to_date(-1)
        ''
    """
    import locale
    from datetime import datetime
    
    if timestamp_ms == -1: return ""
    
    try:
        locale.setlocale(locale.LC_TIME, "es_ES.utf8")  # Try setting locale to Spanish
        dt = datetime.fromtimestamp(timestamp_ms / 1000)
        return dt.strftime("%d de %B de %Y")
    except locale.Error:
        # Fallback to manual month conversion if locale setting fails
        months = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
                  "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
        dt = datetime.fromtimestamp(timestamp_ms / 1000)
        return f"{dt.day} de {months[dt.month - 1]} de {dt.year}"



def format_registration_for_url(registration_number):
    """
    Format registration number for URL construction based on its format
    
    Args:
        registration_number (str): The registration number to format
        
    Returns:
        str: The formatted registration number for URL
    """
    # Case for numbers containing slashes like EU/2/24/320/001-002
    if '/' in registration_number and registration_number.startswith('EU'):
        # Replace '/' with '@' according to the observed pattern
        reg_formatted = registration_number.replace('/', '@')
        return reg_formatted
    else:
        # For cases like "3884 ESP" or "EU033 IP"
        return registration_number.replace(" ", "+")

def format_registration_number(filename):
    """
    Converts a filename into a properly formatted registration number.
    
    Args:
        filename (str): The original filename.
        
    Returns:
        str: The formatted registration number.
    """
    if "-" in filename:
        # Replace first three hyphens with slashes, keeping additional hyphens
        parts = filename.split("-")
        formatted = "/".join(parts[:4])
        if len(parts) > 4:
            formatted += "/" + "-".join(parts[4:])
        return formatted
    
    # Replace underscore with space for other cases
    return filename.replace("_", " ")

def remove_accents(text: str) -> str:
    """
    Remove accents and diacritical marks from text.
    
    Args:
        text: Text to process
        
    Returns:
        Text with accents and diacritics removed
    """
    replacements = {
        'á': 'a', 'à': 'a', 'ä': 'a', 'â': 'a', 'ã': 'a',
        'é': 'e', 'è': 'e', 'ë': 'e', 'ê': 'e',
        'í': 'i', 'ì': 'i', 'ï': 'i', 'î': 'i',
        'ó': 'o', 'ò': 'o', 'ö': 'o', 'ô': 'o', 'õ': 'o',
        'ú': 'u', 'ù': 'u', 'ü': 'u', 'û': 'u',
    #    'ñ': 'n', 'ç': 'c', 
        'ß': 'ss',
        'l·l': 'll', 'ŀl': 'll',
        'ý': 'y', 'ÿ': 'y',
        'ś': 's', 'ź': 'z', 'ż': 'z'
    }
    
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
        # Also replace uppercase
        text = text.replace(orig.upper(), repl.upper())
    
    return text





























