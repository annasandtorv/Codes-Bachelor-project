import re
import yaml
import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt

TPR = "productionA_2025.tpr"
XTC = "productionAmber.xtc"
MAPPING = "MappingDPPC_Amber.yaml"

OUT_DAT = "sn2_order.dat"
OUT_PNG = "sn2_order.png"

u = mda.Universe(TPR, XTC)

with open(MAPPING) as f:
    mapping = yaml.safe_load(f)

z = np.array([0.0, 0.0, 1.0])

# In this DPPC AMBER system, residue pattern is:
# PA PC PA / PA PC PA / ...
# sn-2 = second PA in each lipid block
sn2_pa_residues = [
    res for i, res in enumerate(u.residues)
    if i % 3 == 2 and res.resname == "PA"
]

print("sn-2 PA residues:", len(sn2_pa_residues))


def fix_resname(resname):
    # Mapping may call sn-2 chain OL, but trajectory uses PA for both chains
    if resname == "OL":
        return "PA"
    return resname


pairs = []

# Build sn-2 C-H pairs from mapping
for key, info in mapping.items():
    m = re.match(r"M_G2C(\d+)_M$", key)
    if not m:
        continue

    cnum_universal = int(m.group(1))

    # Skip carbonyl carbon: no C-H bonds
    if cnum_universal < 3:
        continue

    label_num = cnum_universal - 1

    cname = info["ATOMNAME"]
    cres = fix_resname(info["RESIDUE"])

    for hindex in [1, 2, 3]:
        hkey = f"M_G2C{cnum_universal}H{hindex}_M"

        if hkey not in mapping:
            continue

        hname = mapping[hkey]["ATOMNAME"]
        hres = fix_resname(mapping[hkey]["RESIDUE"])

        label = f"{label_num}[{hindex}]"
        pairs.append((label, cres, cname, hres, hname))


print("Number of sn-2 C-H pairs:", len(pairs))
print("First pairs:", pairs[:5])

results = {label: [] for label, *_ in pairs}

for ts in u.trajectory:
    # Skip first 10 ns = 10000 ps
    if ts.time < 10000:
        continue

    for label, cres, cname, hres, hname in pairs:

        # For acyl chains, use only sn-2 PA residues
        if cres == "PA":
            C_atoms = []
            for res in sn2_pa_residues:
                C_atoms.extend(res.atoms.select_atoms(f"name {cname}"))
        else:
            C_atoms = u.select_atoms(f"resname {cres} and name {cname}")

        vals = []

        for c in C_atoms:
            H_atoms = c.residue.atoms.select_atoms(f"name {hname}")

            if len(H_atoms) != 1:
                continue

            vec = H_atoms[0].position - c.position
            norm = np.linalg.norm(vec)

            if norm == 0:
                continue

            vec = vec / norm
            cos_theta = np.dot(vec, z)
            S = 0.5 * (3 * cos_theta**2 - 1)

            vals.append(S)

        if vals:
            results[label].append(np.mean(vals))


labels = []
means = []
stds = []
sems = []

with open(OUT_DAT, "w") as out:
    out.write("# label mean std sem\n")

    for label in results:
        arr = np.array(results[label], dtype=float)

        if len(arr) == 0:
            print(f"WARNING: no values for {label}")
            continue

        mean = np.mean(arr)
        std = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
        sem = std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0

        labels.append(label)
        means.append(mean)
        stds.append(std)
        sems.append(sem)

        out.write(f"{label} {mean:.6f} {std:.6f} {sem:.6f}\n")


if len(labels) == 0:
    raise RuntimeError("No sn-2 order parameters were calculated.")

x = np.arange(len(labels))

plt.figure(figsize=(16, 6))
plt.errorbar(
    x,
    means,
    yerr=sems,
    fmt="o",
    color="blue",
    ecolor="blue",
    capsize=4,
    linewidth=2,
    label="AMBER"
)

plt.xticks(x, labels, rotation=45)
plt.xlabel("sn-2 carbon")
plt.ylabel("Order parameter")
plt.title("Order Parameters - DPPC - sn-2 - AMBER")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(OUT_PNG, dpi=300)
plt.show()

print(f"Saved data: {OUT_DAT}")
print(f"Saved plot: {OUT_PNG}")
