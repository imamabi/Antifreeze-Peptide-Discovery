#!/usr/bin/env python3
"""
Plot GROMACS analysis results across AFP1–AFP6 with correct handling.
- RMSD: 2 columns (time in ps, RMSD) 
- MSD: 3 columns (time in ps, water MSD, protein MSD)
- RMSF: 2 columns (residue index, RMSF) 
- H-bonds: 2 columns (time in ps, count)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches # Added for custom legend

# --- Global style ---
sns.set_style("ticks")  # no grid
sns.set_context("talk", font_scale=1.5)  # larger fonts

# Font / style
plt.rc('font', weight='bold', size=10)

afp_dirs = [f"AFP{i}" for i in range(1,7)]

# --- UPDATED: Use your custom color list ---
system_colors = [
    "steelblue",    # softer than "blue", more professional
    "mediumseagreen",  # clearer than "teal"
    "mediumpurple",    # consistent brightness
    "tomato",          # brighter than "coral"
    "saddlebrown",     # richer than "brown"
    "crimson"          # more distinct than generic "red"
]

# --- UPDATED: Create dark and light shades ---
# User requested: lighter for hydrophobic, "thicker" (darker) for hydrophilic
light_shades = [sns.light_palette(c, n_colors=3, as_cmap=False)[1] for c in system_colors]
dark_shades = [sns.light_palette(c, n_colors=3, as_cmap=False, reverse=True)[0] for c in system_colors]

# Generic colors for the new custom legends
group_colors = ["#C0C0C0", "#606060"] # Light gray (Hydrophobic), Dark gray (Hydrophilic)


def read_xvg(path):
    """Read .xvg file into numpy array, skipping headers."""
    data = []
    with open(path) as f:
        for line in f:
            if line.startswith(("#","@")):
                continue
            parts = line.strip().split()
            if parts:
                data.append([float(x) for x in parts])
    return np.array(data)

# --- RMSD ---
plt.figure(figsize=(8,5))
for afp,c in zip(afp_dirs,system_colors):
    path = os.path.join(afp,"rmsd_backbone.xvg")
    if os.path.exists(path):
        arr = read_xvg(path)
        time_ns = arr[:,0]/1000.0
        plt.plot(time_ns, arr[:,1], label=afp, color=c, lw=2)
plt.xlabel("Time (ns)", fontsize=18)
plt.ylabel("RMSD (nm)", fontsize=18)
# plt.title("Backbone RMSD", fontsize=20, weight="bold")
plt.xlim(0,400)
plt.tick_params(axis='both', labelsize=18) # Increased font size
plt.legend(fontsize=16) # Increased font size
plt.tight_layout()
plt.savefig("rmsd_compare.png", dpi=300)

# --- MSD (protein column = col 3) ---
plt.figure(figsize=(8,5))
for afp,c in zip(afp_dirs,system_colors):
    path = os.path.join(afp,"msd_peptide.xvg")
    if os.path.exists(path):
        arr = read_xvg(path)
        if arr.shape[1] >= 3:
            time_ns = arr[:,0]/1000.0
            plt.plot(time_ns, arr[:,2], label=afp, color=c, lw=2)
plt.xlabel("Time (ns)", fontsize=18)
plt.ylabel("Protein MSD (nm$^2$)", fontsize=18)
# plt.title("Protein MSD", fontsize=20, weight="bold")
plt.xlim(0,350)
plt.tick_params(axis='both', labelsize=18) # Increased font size
plt.legend(fontsize=16) # Increased font size
plt.tight_layout()
plt.savefig("msd_compare.png", dpi=300)

# --- RMSF ---
plt.figure(figsize=(8,5))
for afp,c in zip(afp_dirs,system_colors):
    path = os.path.join(afp,"rmsf_calpha.xvg")
    if os.path.exists(path):
        arr = read_xvg(path)
        plt.plot(arr[:,0], arr[:,1], label=afp, color=c, lw=2)
plt.xlabel("Residue Index", fontsize=18)
plt.ylabel("RMSF (nm)", fontsize=18)
# plt.title("Cα RMSF", fontsize=20, weight="bold")
plt.tick_params(axis='both', labelsize=18) # Increased font size
plt.legend(fontsize=16) # Increased font size
plt.tight_layout()
plt.savefig("rmsf_compare.png", dpi=300)


# --- SASA Distribution ---
plt.figure(figsize=(9,6))
sasa_stats = {}
for afp,c in zip(afp_dirs,system_colors):
    path = os.path.join(afp,"sasa_peptide.xvg")
    if os.path.exists(path):
        arr = read_xvg(path)
        if arr.shape[1] >= 2:
            # Column 1 = time (ps), Column 2 = total SASA (nm^2)
            sasa_vals = arr[:,1]   # use total SASA
            # Plot distribution (KDE + histogram)
            sns.histplot(sasa_vals, kde=True, stat="density",
                         bins=40, color=c, label=afp, alpha=0.4)
            # Compute statistics
            sasa_stats[afp] = {
                "mean": np.mean(sasa_vals),
                "std": np.std(sasa_vals),
                "min": np.min(sasa_vals),
                "max": np.max(sasa_vals)
            }
# Axis labels and title
plt.xlabel("Total SASA (nm$^2$)", fontsize=22, fontweight='bold')
plt.ylabel("Density", fontsize=22, fontweight='bold')
# plt.title("Distribution of SASA Across AFPs", fontsize=20, weight="bold")
plt.tick_params(axis='both', labelsize=20) # Increased font size
plt.legend(fontsize=16, loc='upper right', ncol=3) # Increased font size
plt.tight_layout()
np.savetxt("sasa_stats.txt",
           ["AFP\tMean(nm^2)\tStd(nm^2)\tMin(nm^2)\tMax(nm^2)"] +
           [f"{afp}\t{stats['mean']:.4f}\t{stats['std']:.4f}\t{stats['min']:.4f}\t{stats['max']:.4f}"
            for afp, stats in sasa_stats.items()],
           fmt="%s")
plt.savefig("sasa_distribution.png", dpi=300)

# Print statistics
for afp, stats in sasa_stats.items():
    print(f"{afp}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, "
          f"Min={stats['min']:.2f}, Max={stats['max']:.2f}")

print("\nGenerating H-bond and SASA grouped barplots...")

# --- H-bonds (mean ± std) ---
means, stds, labels, cols = [], [], [], []
for afp,c in zip(afp_dirs,system_colors):
    path = os.path.join(afp,"hbond_sol.xvg")
    if os.path.exists(path):
        arr = read_xvg(path)
        means.append(arr[:,1].mean())
        stds.append(arr[:,1].std())
        labels.append(afp)
        cols.append(c)
if means:
    plt.figure(figsize=(8,5))
    x = np.arange(len(labels))
    # --- UPDATED: Plot hollow bars (outline only) ---
    plt.bar(x, means, yerr=stds, capsize=5, width=0.5, 
            color='none', edgecolor=cols, linewidth=2.5)
    plt.xticks(x, labels, fontsize=20) # Increased font size
    plt.ylabel("Avg. H-bonds", fontsize=22, fontweight='bold')
    plt.xlabel("System", fontsize=22, fontweight='bold')
    # plt.title("Protein–Ice H-bonds (mean ± SD)", fontsize=20, weight="bold")
    plt.tick_params(axis='y', labelsize=20) # Increased font size
    plt.tight_layout()
    plt.savefig("hbonds_compare.png", dpi=300)

for afp, mean, std in zip(labels, means, stds):
    print(f"{afp} H-bonds: Mean={mean:.2f}, Std={std:.2f}\n*3")

np.savetxt("hbond_stats.txt",
           ["System\tMean_Hbonds\tStd_Hbonds"] +
           [f"{afp}\t{mean:.4f}\t{std:.4f}"
            for afp, mean, std in zip(labels, means, stds)],
           fmt="%s")

# --- NEW: Grouped Barplot SASA (Phobic vs Philic) ---
# This replaces the boxplot for a cleaner, publication-ready look
# that is consistent with the H-bond plot.

hydrophobic_sasa_avg = []
hydrophilic_sasa_avg = []
hydrophobic_sasa_std = []
hydrophilic_sasa_std = []

labels_sasa = []
light_colors_sasa = [] # List for bar colors
dark_colors_sasa = []  # List for bar colors
system_to_index_sasa = {afp: i for i, afp in enumerate(afp_dirs)} # Map name to index

for afp in afp_dirs: # Use afp_dirs to keep order
    path_phobic = os.path.join(afp, "sasa_Phobic.xvg")
    path_philic = os.path.join(afp, "sasa_Phillic.xvg")
    
    if os.path.exists(path_phobic) or os.path.exists(path_philic):
        labels_sasa.append(afp)
        color_index = system_to_index_sasa.get(afp, 0) # Find correct color index
        light_colors_sasa.append(light_shades[color_index]) # Add correct light color
        dark_colors_sasa.append(dark_shades[color_index])   # Add correct dark color
        
        if os.path.exists(path_phobic):
            arr_phobic = read_xvg(path_phobic)
            hydrophobic_sasa_avg.append(arr_phobic[:,2].mean())
            hydrophobic_sasa_std.append(arr_phobic[:,2].std())
        else:
            hydrophobic_sasa_avg.append(0)
            hydrophobic_sasa_std.append(0)

        if os.path.exists(path_philic):
            arr_philic = read_xvg(path_philic)
            hydrophilic_sasa_avg.append(arr_philic[:,2].mean())
            hydrophilic_sasa_std.append(arr_philic[:,2].std())
        else:
            hydrophilic_sasa_avg.append(0)
            hydrophilic_sasa_std.append(0)

if labels_sasa:
    plt.figure(figsize=(8,5))
    x = np.arange(len(labels_sasa))
    width = 0.35
    
    # --- UPDATED: Plot bars as hollow outlines ---
    plt.bar(x - width/2, hydrophobic_sasa_avg, width, yerr=hydrophobic_sasa_std, capsize=5, 
            label='Hydrophobic (Light)',
            facecolor='none',                  # Make bar hollow
            edgecolor=dark_colors_sasa,   # Use list of light colors for outlines
            linewidth=2.5)                     # Make outline visible
    
    # --- UPDATED: Plot bars as hollow, hatched outlines ---
    plt.bar(x + width/2, hydrophilic_sasa_avg, width, yerr=hydrophilic_sasa_std, capsize=5,
            label='Hydrophilic (Dark)',
            facecolor='none',                  # Make bar hollow
            edgecolor=dark_colors_sasa,    # Use list of dark colors for outlines
            hatch='//',                         # Add hatching
            linewidth=2.5)                     # Make outline visible
    
    plt.xlabel("System", fontsize=22, fontweight='bold')
    plt.ylabel("Average SASA (nm$^2$)", fontsize=22, fontweight='bold')
    plt.xticks(x, labels_sasa, fontsize=20, rotation=45, ha='right') # Increased font size
    plt.tick_params(axis='y', labelsize=20) # Increased font size

    # --- UPDATED: Use the same generic dark/light legend, but match the new style ---
    light_patch = mpatches.Patch(
        facecolor='none',                  # Hollow
        edgecolor=group_colors[0],         # Light gray outline
        linewidth=1.5,                     # Visible outline
        label='Hydrophobic'
    )
    dark_patch = mpatches.Patch(
        facecolor='none',                  # Hollow
        edgecolor=group_colors[1],         # Dark gray outline
        hatch='//',                         # Match hatching
        linewidth=1.5,                     # Visible outline
        label='Hydrophilic'
    )
    plt.legend(handles=[light_patch, dark_patch], fontsize=16) # Increased font size
    
    plt.tight_layout()
    # --- UPDATED: New filename to reflect new style ---
    plt.savefig("grouped_sasa_bar_hatched.png", dpi=300)

for afp, avg, std in zip(labels_sasa, hydrophobic_sasa_avg, hydrophobic_sasa_std):
    print(f"{afp} Hydrophobic SASA: Mean={avg:.2f}, Std={std:.2f}")

for afp, avg, std in zip(labels_sasa, hydrophilic_sasa_avg, hydrophilic_sasa_std):
    print(f"{afp} Hydrophilic SASA: Mean={avg:.2f}, Std={std:.2f}")

np.savetxt("grouped_sasa_stats.txt",
           ["System\tHydrophobic_Mean(nm^2)\tHydrophobic_Std(nm^2)\t"
            "Hydrophilic_Mean(nm^2)\tHydrophilic_Std(nm^2)"] +
           [f"{afp}\t{hphobic_avg:.4f}\t{hphobic_std:.4f}\t"
            f"{hphilic_avg:.4f}\t{hphilic_std:.4f}"
            for afp, hphobic_avg, hphobic_std, hphilic_avg, hphilic_std in
            zip(labels_sasa, hydrophobic_sasa_avg, hydrophobic_sasa_std,
                hydrophilic_sasa_avg, hydrophilic_sasa_std)],
           fmt="%s")


# --- UPDATED: Grouped H-bonds (Phobic vs Philic) ---
hydrophobic_hbond_avg = []
hydrophilic_hbond_avg = []
labels_hb = []
light_colors_hb = [] # List for bar colors
dark_colors_hb = []  # List for bar colors
system_to_index = {afp: i for i, afp in enumerate(afp_dirs)} # Map name to index

for afp in afp_dirs: # Use afp_dirs to keep order
    # --- CORRECTED: Look for correct filenames ---
    path_phobic = os.path.join(afp, "hbond_protein_ice.xvg")
    path_philic = os.path.join(afp, "hbond_sol.xvg")
    
    if os.path.exists(path_phobic) or os.path.exists(path_philic):
        labels_hb.append(afp)
        color_index = system_to_index.get(afp, 0) # Find correct color index
        light_colors_hb.append(light_shades[color_index]) # Add correct light color
        dark_colors_hb.append(dark_shades[color_index])   # Add correct dark color
        
        if os.path.exists(path_phobic):
            arr_phobic = read_xvg(path_phobic)
            # --- CORRECTED: Read from column 1 (index 1) ---
            hydrophobic_hbond_avg.append(arr_phobic[:,1].mean())
        else:
            hydrophobic_hbond_avg.append(0)

        if os.path.exists(path_philic):
            arr_philic = read_xvg(path_philic)
            # --- CORRECTED: Read from column 1 (index 1) ---
            hydrophilic_hbond_avg.append(arr_philic[:,1].mean())
        else:
            hydrophilic_hbond_avg.append(0)

if labels_hb:
    plt.figure(figsize=(8,5))
    x = np.arange(len(labels_hb))
    width = 0.35
    
    # --- CORRECTED: Use the dynamically built color lists ---
    plt.bar(x - width/2, hydrophobic_hbond_avg, width, label='Backbone-Sol', color=light_colors_hb, edgecolor=dark_colors_hb)
    plt.bar(x + width/2, hydrophilic_hbond_avg, width, label='SideChain-Sol', facecolor='none', edgecolor=dark_colors_hb)
    
    plt.xlabel("System", fontsize=22, fontweight='bold')
    plt.ylabel("Average H-bonds", fontsize=22, fontweight='bold')
    plt.xticks(x, labels_hb, fontsize=20, rotation=45, ha='right') # Increased font size
    plt.tick_params(axis='y', labelsize=20) # Increased font size

    # --- CORRECTED: Create a generic dark/light legend ---
    # This explains the *shading scheme* (light/dark) without
    # conflicting with the *system colors* (blue, green, etc.)
    light_patch = mpatches.Patch(color='none', label='Backbone-Sol')
    dark_patch = mpatches.Patch(color=group_colors[1], label='SideChain-Sol')
    plt.legend(handles=[light_patch, dark_patch], fontsize=16) # Increased font size
    
    plt.tight_layout()
    plt.savefig("grouped_hbond_compare.png", dpi=300)

print("\nAll plots generated.")

