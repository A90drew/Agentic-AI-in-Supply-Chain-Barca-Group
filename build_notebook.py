"""Build InventoryReplenishmentAgent.ipynb from the percent-format script inventory_agent.py.

Markdown cells come from `# %% [markdown]` blocks; `# {{include:file.md}}` pulls in a markdown file as its own cell.
"""
import re, sys
import nbformat as nbf

src = open("inventory_agent.py").read().splitlines()
cells, cur, kind = [], [], None


def flush():
    global cur, kind
    body = "\n".join(cur).strip("\n")
    if body.strip() and kind:
        if kind == "md":
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
            cells.append(nbf.v4.new_markdown_cell(body))
        else:
            cells.append(nbf.v4.new_code_cell(body))
    cur, kind = [], None


for line in src:
    inc = re.match(r"# \{\{include:(.+)\}\}", line)
    if inc:
        flush()
        cells.append(nbf.v4.new_markdown_cell(open(inc.group(1)).read().strip()))
    elif line.startswith("# %% [markdown]"):
        flush(); kind = "md"
    elif line.startswith("# %%"):
        flush(); kind = "code"
    else:
        cur.append(line)
flush()

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "InventoryReplenishmentAgent.ipynb")
print(len(cells), "cells written")
