"""A script to clean up latex files for arxiv submission.

Checks whether there are any unused figures or citations and displays them so
they can be removed. Optionally, the script will remove unused figures for you.
Does not check for commented text. Default folder for figures is 'media'.

Usage:

    python clean_for_arxiv.py [PATH_TO_LATEX_DIR]

"""

import os
import sys
import re

FIG_SUBDIR = "media"

if len(sys.argv) < 2:
    print("Usage:")
    print("\tpython clean_for_arxiv.py [PATH_TO_LATEX_DIR]\n")
    sys.exit()

SRC_PATH = sys.argv[1]

if not os.path.exists(SRC_PATH):
    print("The path", SRC_PATH, "does not exist.")
    sys.exit()

if not os.path.exists(os.path.join(SRC_PATH, FIG_SUBDIR)):
    print("The figure path", os.path.join(SRC_PATH, FIG_SUBDIR), "does not exist.")
    sys.exit()


def recurse_dir(path):
    files = []
    for ele in os.listdir(path):
        if os.path.isdir(os.path.join(path, ele)):
            files += recurse_dir(os.path.join(path, ele))
        else:
            files.append(os.path.join(path, ele))

    return files


all_files = recurse_dir(SRC_PATH)

tex_files = []
bib_files = []
img_files = {}

# Sort the files.
for f in all_files:
    if f.endswith(".tex"):
        tex_files.append(f)
    elif f.endswith(".bib"):
        bib_files.append(f)
    elif f.startswith(os.path.join(SRC_PATH, FIG_SUBDIR)):
        f = f.replace(SRC_PATH, "")
        f = f[1:]
        f = f.replace("\\", "/").strip()
        img_files[f] = False

# Get the list of citation keys.
citation_keys = {}
for bib in bib_files:
    with open(bib, 'r') as f:
        for line in f.readlines():
            line = line.strip()

            m = re.search(r'^@.*{(.*),', line)
            if m:
                citation_keys[m.group(1).strip()] = False

# Check each tex file for the keys and images to get missing ones.
for tex in tex_files:
    print("Checking", tex)
    with open(tex, 'r') as f:
        for line in f.readlines():
            line = line.strip()

            # Check for images.
            m = re.search(r'\\includegraphics.*{(.*)}', line)
            if m:
                if m.group(1).strip() not in img_files.keys():
                    print("Panic!", m.group(1), "not in keys")
                img_files[m.group(1).strip()] = True

            # Check for citations.
            m = re.findall(r'\\cite{([^}]*)', line)
            if len(m) > 0:
                m = ",".join(m)
                keys = [k.strip() for k in m.split(",")]
                for k in keys:
                    if k not in citation_keys.keys():
                        print("Panic!", k, "not in keys")
                    citation_keys[k] = True

unused_citations = []
for k, v in citation_keys.items():
    if not v:
        unused_citations.append(k)

print()
if len(unused_citations) == 0:
    print("No unused citations.")
else:
    print("The following citations are unused:")
    print("\t" + "\n\t".join(unused_citations))

unused_figs = []
for k, v in img_files.items():
    if not v:
        unused_figs.append(k)
        print("\t", k)

print()
if len(unused_figs) == 0:
    print("No unused figures.")
else:
    print("The following figures are unused:")
    print("\t" + "\n\t".join(unused_figs))

    print("Would you like to remove them? [y/N]")
    resp = input()
    if resp.strip().lower() == "y":
        for fig in unused_figs:
            os.remove(os.path.join(SRC_PATH, fig))
        print("Removed unused figures.")
    else:
        print("Unused figures were not removed.")
