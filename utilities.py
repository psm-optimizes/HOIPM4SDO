import numpy as np
import scipy as sp
import pandas as pd
from scipy import linalg
from copy import deepcopy

import math
import time
import random

def is_pos_def(x):
  return np.all(np.linalg.eigvals(x) > 0)

def hvec(A):
	n=len(A)
	s=[]
	for i in range(n):
		for j in range(n):
			s.append(A[i,j])
	return np.array(s)

def hmat(s):
  n=int(np.sqrt(len(s)))
  A=np.zeros((n,n))
  t=0
  for i in range(n):
    for j in range(n):
      A[i,j]=s[t]
      t=t+1
  return A

def svec(A):
  n=len(A)
  s=[]
  for i in range(n):
    s.append(A[i,i])
    for j in range(i+1,n):
      s.append(np.sqrt(2)*A[i,j])
  return np.array(s)

def smat(s):
  n=int(np.floor(np.sqrt(2*len(s))))
  A=np.zeros((n,n))
  t=0
  for i in range(n):
    A[i,i]=s[t].item()
    t=t+1
    for j in range(i+1,n):
      A[i,j]=s[t].item()/np.sqrt(2)
      A[j,i]=s[t].item()/np.sqrt(2)
      t=t+1
  return A

def invsqrt(B):
  return np.linalg.inv(sp.linalg.sqrtm(B))

def symet(M,type):
	if type=="AHO":
		return 0.5*(M+M.T)
	elif type=="NT":
		# Dmid = np.linalg.cholesky(np.matmul(np.matmul(np.linalg.cholesky(S),X),np.linalg.cholesky(S)))
		# D = np.matmul(np.matmul(invsqrt(S),Dmid), invsqrt(S))
		Wmid = invsqrt(np.matmul(np.matmul(np.linalg.cholesky(X),S),np.linalg.cholesky(X)))
		W = np.matmul(np.matmul(np.linalg.cholesky(X),Wmid), np.linalg.cholesky(X))
		P = np.linalg.cholesky(W)
		return 0.5*(np.matmul(np.matmul(P,M),np.linalg.inv(P)) + np.transpose(np.matmul(np.matmul(P,M),np.linalg.inv(P))))
	elif type=="HRV":
		return 0.5*(np.matmul(np.matmul(np.linalg.cholesky(S),M),invsqrt(S)) + np.transpose(np.matmul(np.matmul(np.linalg.cholesky(S),M),invsqrt(S))))
	elif type=="KSS":
		return 0.5*(np.matmul(np.matmul(invsqrt(X),M),np.linalg.cholesky(X)) + np.transpose(np.matmul(np.matmul(invsqrt(X),M),np.linalg.cholesky(X))))
     
'''
svec Implementation Functions
'''
def index_list(A):
  n = len(A)
  idx_lst=[]
  for i in range(1,n+1):
    idx_lst.append((i,i))
    for j in range(i+1,n+1):
      idx_lst.append((i,j))
  return np.array(idx_lst)

def i_idx(i,j,n):
  return (i-1)*n + j

def j_idx(i,j,n):
  return (j-1)*n + i

def symkron(G,K):
  nkron = np.kron(G,K).shape[0]
  n  = int(np.sqrt(nkron))
  ns = int(0.5*n*(n+1))
  Q = np.zeros((ns,nkron))

  for itrt in range(len(index_list(np.eye(n)))):
    i,j = index_list(np.eye(n))[itrt]
    if (i == j):
      Q[itrt, i_idx(i,j,n)-1] = 1
    else:
      Q[itrt, i_idx(i,j,n)-1] = 1/np.sqrt(2)
      Q[itrt, j_idx(i,j,n)-1] = 1/np.sqrt(2)

  return 0.5*np.matmul(np.matmul(Q,(np.kron(G,K)+np.kron(K,G))),Q.T)

def forward_substitution(L, b):

    #Get number of rows
    n = L.shape[0]

    #Allocating space for the solution vector
    y = np.zeros_like(b, dtype=np.double);

    #Here we perform the forward-substitution.
    #Initializing  with the first row.
    y[0] = b[0] / L[0, 0]

    #Looping over rows in reverse (from the bottom  up),
    #starting with the second to last row, because  the
    #last row solve was completed in the last step.
    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i,:i], y[:i])) / L[i,i]

    return y


def back_substitution(U, y):

    #Number of rows
    n = U.shape[0]

    #Allocating space for the solution vector
    x = np.zeros_like(y, dtype=np.double);

    #Here we perform the back-substitution.
    #Initializing with the last row.
    x[-1] = y[-1] / U[-1, -1]

    #Looping over rows in reverse (from the bottom up),
    #starting with the second to last row, because the
    #last row solve was completed in the last step.
    for i in range(n-2, -1, -1):
        x[i] = (y[i] - np.dot(U[i,i:], x[i:])) / U[i,i]

    return x

'''
Reporting Tools
'''
def draw_table(data):
    # Get the number of rows and columns based on the input data
    rows = len(data)
    cols = len(data[0]) if rows > 0 else 0

    # Calculate the width of each column (assuming fixed-width columns for simplicity)
    col_widths = [max(len(str(data[row][col])) for row in range(rows)) for col in range(cols)]

    # Function to draw the separator line (row divider)
    def draw_separator():
        return '+' + '+'.join(['-' * (width + 2) for width in col_widths]) + '+'

    # Function to draw a row of data
    def draw_row(row_data):
        return '|' + '|'.join([f' {str(row_data[i]).ljust(col_widths[i])} ' for i in range(cols)]) + '|'

    # Build the full table as a string
    table = [draw_separator()]
    for row in data:
        table.append(draw_row(row))
        table.append(draw_separator())

    # Join the table lines and print it
    return '\n'.join(table)

def records_to_latex_table(
    Records: dict,
    caption: str | None = None,
    label: str | None = None,
    float_spec: str = "H",
    ) -> str:
    """
    Convert Records dict like {(rho,p): [[header...],[row...],...], ...}
    into a combined LaTeX tabular table like the example.

    - Assumes each value is a list-of-rows where first row is a header.
    - Pads missing iterations with '-' and replaces None ratio with 'N/A'.
    - Converts 'e' scientific notation to LaTeX-friendly 'E' (as in example).
    """

    def fmt(x):
        if x is None:
            return "N/A"
        if isinstance(x, str):
            s = x.strip()
            # normalize scientific notation: 1.00e+00 -> 1.00E+00
            if "e" in s:
                s = s.replace("e", "E")
            return s
        return str(x)

    # Sort keys for stable column order (e.g., (2,2), (2,4), (2,8))
    keys = sorted(Records.keys(), key=lambda t: (t[0], t[1]))

    # Build a per-key mapping: iter -> (mu, ratio)
    per_key = {}
    max_iter = 0
    for k in keys:
        rows = Records[k]
        data_rows = rows[1:]  # skip header
        m = {}
        for r in data_rows:
            it = int(r[0])
            mu = fmt(r[1])
            ratio = fmt(r[2])
            m[it] = (mu, ratio)
            max_iter = max(max_iter, it)
        per_key[k] = m

    # LaTeX lines
    lines = []
    lines.append(f"\\begin{{table}}[{float_spec}]")
    if caption is not None:
        lines.append(f"\\caption{{{caption}}}")
    if label is not None:
        lines.append(f"\\label{{{label}}}")

    # Column spec: 1 (Iter) + 2 per key
    col_spec = "c" * (1 + 2 * len(keys))
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")

    # First header row with multicolumns
    parts = ["~"]
    for (rho, p) in keys:
        parts.append(f"\\multicolumn{{2}}{{c}}{{$({{\\rho}},p)=({rho},{p})$}}")
    lines.append(" & ".join(parts) + " \\\\ \\cline{2-%d}" % (1 + 2 * len(keys)))

    # Second header row
    hdr = ["Iter ($k$)"]
    for _ in keys:
        hdr += ["$\\langle X_k,S_k\\rangle/n$", "Ratio"]
    lines.append(" & ".join(hdr) + " \\\\ \\hline")

    # Data rows 0..max_iter
    for it in range(0, max_iter + 1):
        row = [str(it)]
        for k in keys:
            if it in per_key[k]:
                mu, ratio = per_key[k][it]
            else:
                mu, ratio = "-", "-"
            row += [mu, ratio]
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)