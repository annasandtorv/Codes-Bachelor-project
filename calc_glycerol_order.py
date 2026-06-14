import yaml
import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt

TPR = "productionA_2025.tpr"
XTC = "productionAmber.xtc"
MAPPING = "MappingDPPC_Amber.yaml"

OUT_DAT = "glycerol_order.dat"
OUT_PNG = "glycerol_order.png"

u = mda.Universe(TPR, XTC)

with open(MAPPING) as f:
    mapping = yaml.safe_load(f)

z = np.array([0.0, 0.0, 1.0])

pairs = []

glycerol_atoms = [
    ("M_G1_M", "g1"),
    ("M_G2_M", "g2"),
    ("M_G3_M", "g3"),
]

for carbon_key, label_base in glycerol_atoms:
    info = mapping[carbon_key]

    cname = info["ATOMNAME"]
    cres = info["RESIDUE"]

    if cres in ["PA", "OL"]:
        cres = "MY"

    for hindex in [1, 2]:
        hkey = carbon_key.replace("_M", f"H{hindex}_M")

        if hkey not in mapping:
            continue

        hname = mapping[hkey]["ATOMNAME"]
        hres = mapping[hkey]["RESIDUE"]

        if hres in ["PA", "OL"]:
            hres = "MY"

        label = f"{label_base}[{hindex}]"
        pairs.append((label, cres, cname, hres, hname))


results = {label: [] for label, *_ in pairs}

for ts in u.trajectory:
    if ts.time < 10000:
        continue

    for label, cres, cname, hres, hname in pairs:
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

        if len(arr) > 1:
            std = np.std(arr, ddof=1)
            sem = std / np.sqrt(len(arr))
        else:
            std = 0.0
            sem = 0.0

        labels.append(label)
        means.append(mean)
        stds.append(std)
        sems.append(sem)

        out.write(f"{label} {mean:.6f} {std:.6f} {sem:.6f}\n")


x = np.arange(len(labels))

plt.figure(figsize=(9, 5))

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
plt.xlabel("Glycerol segment")
plt.ylabel("Order parameter")
plt.title("Order Parameters - DPPC - glycerol backbone - AMBER")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(OUT_PNG, dpi=300)
plt.show()

print(f"Saved data: {OUT_DAT}")
print(f"Saved plot: {OUT_PNG}")
