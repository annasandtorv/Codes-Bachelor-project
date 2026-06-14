import yaml
import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt

TPR = "productionA_2025.tpr"
XTC = "productionAmber.xtc"
MAPPING = "MappingDPPC_Amber.yaml"

OUT_DAT = "headgroup_order.dat"
OUT_PNG = "headgroup_order.png"

u = mda.Universe(TPR, XTC)

with open(MAPPING) as f:
    mapping = yaml.safe_load(f)

z = np.array([0.0, 0.0, 1.0])


def fix_resname(resname):
    if resname in ["PA", "OL"]:
        return "MY"
    return resname


def calc_pair_timeseries(carbon_key, hindex):
    """
    Calculates one OP time series for one mapped C-H pair.
    Returns one averaged value per frame.
    """
    hkey = carbon_key.replace("_M", f"H{hindex}_M")

    if carbon_key not in mapping or hkey not in mapping:
        return []

    c_info = mapping[carbon_key]
    h_info = mapping[hkey]

    cname = c_info["ATOMNAME"]
    hname = h_info["ATOMNAME"]

    cres = fix_resname(c_info["RESIDUE"])
    hres = fix_resname(h_info["RESIDUE"])

    values = []

    for ts in u.trajectory:
        # skip first 10 ns = 10000 ps
        if ts.time < 10000:
            continue

        C_atoms = u.select_atoms(f"resname {cres} and name {cname}")

        frame_vals = []

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
            frame_vals.append(S)

        if frame_vals:
            values.append(np.mean(frame_vals))

    return values


# ---------------------------------------------------------
# Headgroup definition with 7 final points:
#
# alpha[1], alpha[2] = M_G3C4 hydrogens
# beta[1], beta[2]   = M_G3C5 hydrogens
# gamma[1-3]         = average over M_G3N6C1, C2, C3
# ---------------------------------------------------------

series = {}

# alpha
series["alpha[1]"] = calc_pair_timeseries("M_G3C4_M", 1)
series["alpha[2]"] = calc_pair_timeseries("M_G3C4_M", 2)

# beta
series["beta[1]"] = calc_pair_timeseries("M_G3C5_M", 1)
series["beta[2]"] = calc_pair_timeseries("M_G3C5_M", 2)

# gamma: average equivalent methyl groups for each H index
gamma_carbons = [
    "M_G3N6C1_M",
    "M_G3N6C2_M",
    "M_G3N6C3_M",
]

for hindex in [1, 2, 3]:
    gamma_series = []

    for carbon_key in gamma_carbons:
        vals = calc_pair_timeseries(carbon_key, hindex)

        if vals:
            gamma_series.append(vals)

    if gamma_series:
        # make all time series same length just in case
        min_len = min(len(v) for v in gamma_series)
        arr = np.array([v[:min_len] for v in gamma_series])

        # average over the three methyl groups for each frame
        series[f"gamma[{hindex}]"] = list(np.mean(arr, axis=0))
    else:
        series[f"gamma[{hindex}]"] = []


labels_order = [
    "alpha[1]",
    "alpha[2]",
    "beta[1]",
    "beta[2]",
    "gamma[1]",
    "gamma[2]",
    "gamma[3]",
]

labels = []
means = []
stds = []
sems = []

with open(OUT_DAT, "w") as out:
    out.write("# label mean std sem\n")

    for label in labels_order:
        arr = np.array(series[label], dtype=float)

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

plt.figure(figsize=(8, 5))

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
plt.xlabel("Headgroup segment")
plt.ylabel("Order parameter")
plt.title("Order Parameters - DPPC - headgroup - AMBER")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(OUT_PNG, dpi=300)
plt.show()

print(f"Saved data: {OUT_DAT}")
print(f"Saved plot: {OUT_PNG}")
