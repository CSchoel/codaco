import re
from typing import *
import unicodedata as ud
import pandas as pd
import io

def replace_inline_tabs(text: str, tabsize=4):
    res = ""
    for l in text.splitlines(keepends=True):
        ioff = len(res)
        for c in l:
            if c == '\t':
                i = len(res) - ioff
                res += ' ' * (tabsize - i % tabsize)
            else:
                res += c
    return res

def simple_table_blocks(text: str) -> List[Tuple[int, int, int]]:
    """
    Finds line spans in a text that are likely to contain a
    fixed-width table.
    """
    lines = text.splitlines()
    edgecount = [len(re.findall(r"\S(\t|  )", l)) for l in lines]
    tablines = [i for i in range(len(lines)) if edgecount[i] > 0]
    tablines.append(max(tablines) + 2) # add gap at end as sentinel
    combined = []
    start = 0 if len(tablines) == 0 else tablines[0]
    for cur,nxt in zip(tablines[:-1], tablines[1:]):
        if (nxt-cur) > 1: # gap
            if (cur-start) > 1: # at least two lines
                combined.append((sum(edgecount[start:cur+1]), start, cur+1))
            start = nxt
    return sorted(combined, reverse=True)

def tableness(text: str):
    """
    Calculates a "tableness" score for a string by counting
    the number of edges (indices where a space is followed
    by a non-space or vice versa) that continue into the
    next line.

    This is a helper function for guess_tabwidth.
    """
    lastedges = set()
    score = 0
    for l in text.splitlines():
        # count number of edges that continue from last line
        edges = {i for i in range(len(l)-1) if l[i].isspace() ^ l[i+1].isspace()}
        score += len(lastedges.intersection(edges))
        lastedges = edges
    return score

def guess_tabwidth(text: str, guesses=[2, 4, 6, 8]):
    """
    Guesses the tab width used for text document containing
    fixed-width tables.

    Uses tableness() function to determine a score based on
    the number of edges (i.e. indices where a space is followed
    by a non-scpae or vice versa) that continue into the next
    line.

    Returns the guess out of guesses that produced the highest
    tableness score.
    """
    scored = [(tableness(replace_inline_tabs(text, g)), g) for g in guesses]
    s, g = max(scored)
    return g


def get_block(text: str, start: int, end: int):
    return "\n".join(x for i, x in enumerate(text.splitlines()) if start <= i < end)


def get_table(text: str, start: int, end: int):
    b = get_block(text, start, end)
    # remove lines that only contain punctuation, symbol and separator chars
    b = "".join(l for l in b.splitlines(keepends=True) if any(ud.category(c)[0] not in ['Z', 'P', 'S'] for c in l))
    return pd.read_fwf(io.StringIO(b))

def sparse_locmax(points: Dict[int, int]) -> Dict[int, int]:
    """
    Finds local maxima in a sparse list
    """
    # Test case 1: {0:3, 1:2, 2:3, 3:2, 4:2, 5:3, 6:2, 7:3} -> {0: 3, 2: 3, 5: 3, 7: 3}
    # Test case 2: {0:1, 1:2, 2:3, 4:3, 5:2, 6:1, 18:3} -> {2: 3, 4: 3, 18: 3}
    maxima = {}
    imax = None
    # insert gap sentinel to make sure last maximum is also found
    sentinel = max(points.keys(), default=0) + 2
    for i in sorted(points.keys()) + [sentinel]:
        if (i-1) not in points:
            # gap -> assume start of new maximum
            if imax is not None:
                maxima[imax] = points[imax]
            imax = i
        elif points[i] > points[i-1]:
            # increase -> update maximum
            imax = i
        elif points[i] < points[i-1]:
            # decrease -> imax was local maximum
            if imax is not None:
                maxima[imax] = points[imax]
            # currently we do not have a candidate for new maximum
            imax = None
        # else: plateau -> do nothing
    return maxima

def select_columns(continuation: Dict[int, int]) -> Dict[int, int]:
    """
    Selects only columns that have a height of at least 3 and
    that are local maxima.

    Helper function to be used in find_table_blocks().
    """
    # only keep columns of at least height 3
    values = filter(lambda x: x[1] > 2, continuation.items())
    # only keep local maxima
    values = sparse_locmax(dict(values))
    return values

def values_in_column(lines: List[str], column: int):
    """
    Scores a potential column by examining its edges and
    counting how many lines of the column are actually
    edges (i.e. transitions between space and non-space
    characters).
    """
    # move to leftmost end of column separator
    leftmost = column
    while leftmost > 0 and all(l[leftmost-1] == ' ' for l in lines):
        leftmost -= 1
    rightmost = column
    while all(len(l) > rightmost + 1 and l[rightmost+1] == ' ' for l in lines):
        rightmost += 1
    lvalues = [False] * len(lines) if leftmost == 0 else (l[leftmost-1] != ' ' for l in lines)
    rvalues = (len(l) > rightmost + 1 and l[rightmost+1] != ' ' for l in lines)
    has_value = (l or r for l,r in zip(lvalues, rvalues))
    return sum(has_value)

def max_cells(lines: List[str], continuation: Dict[int, int]) -> Tuple[int, int]:
    """
    Determines the best candidate for a fixed-width table based on
    column run-length data.
    """
    # successively test how many cells a table of height v
    # would have for each column height v in continuation
    options = []
    cols_by_height = {}
    for k, v in continuation.items():
        cols_by_height.setdefault(v, [])
        cols_by_height[v].append(k)
    # by staring with the highest column, we know that we
    # only have to add more column indices, not substract those
    # that are not high enough
    cbh_sorted = sorted(cols_by_height.items(), reverse=True)
    indices = []
    for v, ks in cbh_sorted:
        indices += ks
        n = sum([values_in_column(lines[len(lines)-v:], idx) for idx in indices])
        options.append((n, v))
    return max(options, default=(0, 0))


def find_table_blocks(text: str) -> List[Tuple[int, int, int]]:
    """
    Finds candidates for fixed-width table blocks in a string by
    generating run-lengths of potential columns and evaluating
    candidates with max_cells().

    Comparison to simple_table_blocks():
        Advantage:
            * Finds tables that have only a single space as column separator.
        Disadvantage:
            * Much more complicated / worse performance.
            * Might currently miss tables where one column is continued after the end of the table.

    Returns:
        (score, start, end): tuple with score, start and end
            index of candidate. End index is exclusive.
    """
    # find longest consecutive number of lines where more than one column consists entirely of spaces
    lastline = {}
    found = []
    lines = text.splitlines()
    lines.append("") # add empty line as sentinel
    for i, l in enumerate(lines):
        colcount = {j: lastline.get(j, 0) + 1 for j, c in enumerate(l) if c == " "}
        # removes leading columns zero, because they stem from indentation
        j = 0
        while j in colcount:
            del colcount[j]
            j += 1
        # check if we are at the end of a consecutive run
        # => i.e. the maximum runlenght of the previous line was higher
        # TODO maybe additional condition: number of columns with maximum value drops
        # pro: would find tables where one column continues after the end of the table
        # con: would report a lot of unfinished tables (need to filter overlap?)
        if max(lastline.values(), default=0) > max(colcount.values(), default=0):
            # no continuing lines found
            if max(lastline.values()) > 2:
                print(i, lastline)
                # colum height is at least 3
                score, height = max_cells(lines[0:i], select_columns(lastline))
                found.append((score, i - height, i))
        lastline = colcount
    return sorted(found, reverse=True)
