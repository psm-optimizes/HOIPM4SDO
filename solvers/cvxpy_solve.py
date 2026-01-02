'''
Code used to call solvers such as MOSEK and SDPA to solve the isntances
'''
import cvxpy as cp
import re
from typing import List, Union, Tuple

def cp_solve(solver,n,m,A,b,C):
    X = cp.Variable((n+1,n+1), symmetric=True)

    constraints = [X >> 0]
    constraints += [
        cp.trace(A[i] @ X) == b[i] for i in range(m+1)
    ]
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)),
                    constraints)

    if solver == 'MOSEK':
        prob.solve(solver=cp.MOSEK, mosek_params={
            'MSK_DPAR_INTPNT_TOL_PFEAS': 1e-14,
            'MSK_DPAR_INTPNT_TOL_DFEAS': 1e-14,
            'MSK_DPAR_INTPNT_TOL_REL_GAP': 1e-10,
            'MSK_DPAR_INTPNT_CO_TOL_MU_RED': 1e-10,
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-10,
            'MSK_DPAR_INTPNT_TOL_DSAFE': 10,
            'MSK_DPAR_INTPNT_TOL_PSAFE': 10
        }, verbose=True)

    elif solver == 'SDPA':
        prob.solve(solver=cp.SDPA, 
                   epsilonStar=1e-10, 
                   lambdaStar=1E-00, 
                   epsilonDash=1e-10, 
                   print='display', 
                   solver_verbose=1, verbose=1, 
                   resultFile='sdpa-log.txt', 
                   sdpaResult='result.txt')

    else:
        raise Exception('Solver not found')

'''
Following functions are defined to facilitate extracting gap
values from the log file of the solvers. 'extract_mu' is for
Mosek and 'extract_mu_sdpa' is for SDPA.
'''


def extract_mu(log_text: str) -> List[float]:
    """
    Extracts the MU column from MOSEK/CVXPY iteration tables inside arbitrary text.
    Works even if lines are prefixed with timestamps like:
      (CVXPY) Sep 01 06:26:38 PM: 0  1.6e+00 ...  1.0e+00  0.03
    """
    mu_values: List[float] = []
    lines = log_text.splitlines()

    # Numeric token (float or int, supports scientific notation)
    num_pat = re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?')

    # Try to find the header first, then parse the rows after it
    header_idx = None
    for i, ln in enumerate(lines):
        if re.search(r'\bITE\b', ln) and re.search(r'\bMU\b', ln) and re.search(r'\bTIME\b', ln):
            header_idx = i
            break

    # Helper: return tokens from the part of the line after the last colon (to skip timestamps)
    def tokens_after_prefix(line: str):
        _, sep, tail = line.rpartition(':')  # split on the last colon only
        segment = tail if sep else line
        return num_pat.findall(segment)

    if header_idx is not None:
        for ln in lines[header_idx + 1:]:
            toks = tokens_after_prefix(ln)
            # Expect at least 9 numeric tokens: ITE PFEAS DFEAS GFEAS PRSTATUS POBJ DOBJ MU TIME
            if len(toks) >= 9:
                try:
                    mu_values.append(float(toks[7]))  # MU is the 8th numeric token
                except ValueError:
                    pass
        if mu_values:
            return mu_values

    # Fallback: pattern-match iteration rows directly (no header needed)
    row_pat = re.compile(
        r'^\s*(?:.*?:\s*)?'            # optional prefix "(...): "
        r'\d+\s+'                      # ITE
        r'[0-9.eE+\-]+\s+'             # PFEAS
        r'[0-9.eE+\-]+\s+'             # DFEAS
        r'[0-9.eE+\-]+\s+'             # GFEAS
        r'[0-9.eE+\-]+\s+'             # PRSTATUS
        r'[0-9.eE+\-]+\s+'             # POBJ
        r'[0-9.eE+\-]+\s+'             # DOBJ
        r'([0-9.eE+\-]+)\s+'           # MU (capture)
        r'[0-9.eE+\-]+\s*$',           # TIME
        re.M
    )
    return [float(x) for x in row_pat.findall(log_text)]


NumOrStr = Union[float, str]
OutType = Union[List[NumOrStr], List[Tuple[int, NumOrStr]]]

def extract_mu_sdpa(text: str, *, as_str: bool = False, fmt: str = ".2e",
                          include_index: bool = False) -> OutType:
    """
    Extract the 'mu' column from a CVXPY/SDPA iteration table such as:

       mu      thetaP  thetaD  objP  objD  alphaP  alphaD  beta
     0 1.0e+04 1.0e+00 ...

    Returns:
      - list of floats (default), or
      - list of strings with given format if as_str=True
      - optionally (index, value) pairs if include_index=True
    """
    # Float or int with optional exponent, allowing leading +/-
    num_pat = re.compile(r'[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?')

    lines = text.splitlines()

    # Find header line (tolerant to spacing/case)
    header_idx = None
    for i, ln in enumerate(lines):
        if (re.search(r'\bmu\b', ln, re.I) and
            re.search(r'\bthetaP\b', ln, re.I) and
            re.search(r'\bbeta\b', ln, re.I)):
            header_idx = i
            break

    start = header_idx + 1 if header_idx is not None else 0
    result: List[Union[float, str, Tuple[int, Union[float, str]]]] = []

    for ln in lines[start:]:
        if not ln.strip():
            continue
        # Ensure the line starts with an integer index
        m_idx = re.match(r'^\s*(\d+)\b', ln)
        if not m_idx:
            continue
        idx = int(m_idx.group(1))

        # Grab all numeric tokens; expect: [index, mu, thetaP, ...]
        toks = num_pat.findall(ln)
        if len(toks) < 2:
            continue

        # Second numeric token after the index is 'mu'
        try:
            mu_val = float(toks[1])
        except ValueError:
            continue

        out_val: NumOrStr = format(mu_val, fmt) if as_str else mu_val
        result.append((idx, out_val) if include_index else out_val)

    return result