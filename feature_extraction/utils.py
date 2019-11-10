import os
import re


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]


def get_dirs(path, prefix='', suffix=''):
    '''
    Returns a sorted list of all directories under 'path'
    @param path :string
    @param prefix :returns all names that start with prefix
    @param suffix :returns all names that end with suffix
    '''
    dirs = os.listdir(path)
    dirs.sort(key=natural_keys)
    items = [f for f in dirs if os.path.isdir(os.path.join(path, f)) and
                                f.startswith(str(prefix)) and f.endswith(str(suffix))]
    return items


def get_files(path, prefix='', suffix=''):
    '''
    Returns a sorted list of all files under 'path'
    @param path :string
    @param prefix :returns all names that start with prefix
    @param suffix :returns all names that end with suffix
    '''
    dirs = os.listdir(path)
    dirs.sort(key=natural_keys)
    items = [f for f in dirs if os.path.isfile(os.path.join(path, f)) and
                                f.startswith(str(prefix)) and f.endswith(str(suffix))]
    return items

