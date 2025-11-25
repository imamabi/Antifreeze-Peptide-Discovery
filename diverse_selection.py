import os, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf  # shrinkage for stability
from scipy.stats import chi2
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# E1..E5 per amino acid (Venkatarajan & Braun, 2001; Table 1)
E_TABLE = {
    "A": [ 0.008,  0.134, -0.475, -0.039,  0.181],
    "R": [ 0.171, -0.361,  0.107, -0.258, -0.364],
    "N": [ 0.255,  0.038,  0.117,  0.118, -0.055],
    "D": [ 0.303, -0.057, -0.014,  0.225,  0.156],
    "C": [-0.132,  0.174,  0.070,  0.565, -0.374],
    "Q": [ 0.149, -0.184, -0.030,  0.035, -0.112],
    "E": [ 0.221, -0.280, -0.315,  0.157,  0.303],
    "G": [ 0.218,  0.562, -0.024,  0.018,  0.106],
    "H": [ 0.023, -0.177,  0.041,  0.280, -0.021],
    "I": [-0.353,  0.071, -0.088, -0.195, -0.107],
    "L": [-0.267,  0.018, -0.265, -0.274,  0.206],
    "K": [ 0.243, -0.339, -0.044, -0.325, -0.027],
    "M": [-0.239, -0.141, -0.155,  0.321,  0.077],
    "F": [-0.329, -0.023,  0.072, -0.002,  0.208],
    "P": [ 0.173,  0.286,  0.407, -0.215,  0.384],
    "S": [ 0.199,  0.238, -0.015, -0.068, -0.196],
    "T": [ 0.068,  0.147, -0.015, -0.132, -0.274],
    "W": [-0.296, -0.186,  0.389,  0.083,  0.297],
    "Y": [-0.141, -0.057,  0.425, -0.096, -0.091],
    "V": [-0.274,  0.136, -0.187, -0.196, -0.299],
}

def parse_fasta(path):
    """Yield (seq_id, sequence) from a FASTA file."""
    header, chunks = None, []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks).upper()
                header = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks).upper()

'''ArithmeticError
def seq_to_E_avg(seq, skip_unknown=True):
    """Average per-residue E1..E5 for a sequence."""
    rows = []
    for aa in seq:
        if aa in E_TABLE:
            rows.append(E_TABLE[aa])
        elif not skip_unknown:
            rows.append([0.0]*5)
    if not rows: return None
    return np.mean(np.array(rows, dtype=float), axis=0)'''

def seq_to_E_stats(seq, skip_unknown=True):
    """
    Compute mean, variance, min, max of each E1..E5 dimension for a sequence.
    Returns a 20-element vector [E1_mean, E2_mean, ..., E5_mean,
                                  E1_var,  E2_var,  ..., E5_var,
                                  E1_min,  E2_min,  ..., E5_min,
                                  E1_max,  E2_max,  ..., E5_max]
    """
    rows = []
    for aa in seq:
        if aa in E_TABLE:
            rows.append(E_TABLE[aa])
        elif not skip_unknown:
            rows.append([0.0]*5)
    if not rows:
        return None

    arr = np.array(rows, dtype=float)  # shape (L, 5)

    means = arr.mean(axis=0)
    #vars_ = arr.var(axis=0)                                     #### You can choose mean and max: mean tends to capture the overall trend and max highlights the strongest signal 
    #mins = arr.min(axis=0)
    #maxs = arr.max(axis=0)

    #return np.concatenate([means, maxs])
    return means

FASTA_PATH = "/path/to/afp/training/dataset"   # <-- set your file

ids, seqs, E_vectors = [], [], []
for sid, seq in parse_fasta(FASTA_PATH):
    v = seq_to_E_stats(seq)
    if v is not None:
        ids.append(sid)
        seqs.append(seq)
        E_vectors.append(v)

df_E = pd.DataFrame(E_vectors, columns=["E1","E2","E3","E4","E5"])
#df_E.insert(0, "Sequence_ID", ids)

df_S= pd.DataFrame({"Sequence_ID": ids,
                    "Sequence": seqs})


df_all = pd.concat([df_S, df_E], axis=1)

df_all = df_all[df_all["Sequence_ID"].str.startswith("AFP")].reset_index(drop=True)

print(f"Sequences retained: {len(df_all)}")
df_all.tail()

# Standardize across sequences
X = df_all.drop(columns=["Sequence_ID", "Sequence"]).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

afp_scaled = X_scaled

# Load your parsed FASTA DataFrame
df = pd.read_csv("filtered_candidates.csv")

E_df = pd.DataFrame(df['sequence'].apply(seq_to_E_stats).tolist(),columns=["E1","E2","E3","E4","E5"])

# Add the 'sequence' column from the original df to E_df
E_df['sequence'] = df['sequence']

# standardize across sequences
X1 = E_df[["E1","E2","E3","E4","E5"]]
X1.values
scaler = StandardScaler()
X1_scaled = scaler.fit_transform(X1)
filtered_scaled = X1_scaled


#Mahalanobis in original (standardized) space
def mahalanobis_scores(X_ref, X_query, use_shrinkage=True):
    """
    X_ref:    reference matrix (AFP), shape (n_ref, n_features)
    X_query:  candidate matrix, shape (n_query, n_features)
    Returns: (d, d2, mu, Sigma_inv)
        d:   Mahalanobis distances, shape (n_query,)
        d2:  squared distances, shape (n_query,)
    """
    if use_shrinkage:
        cov_est = LedoitWolf().fit(X_ref)
        mu = cov_est.location_
        Sigma_inv = np.linalg.pinv(cov_est.covariance_)
    else:
        mu = np.mean(X_ref, axis=0)
        Sigma = np.cov(X_ref, rowvar=False)
        Sigma_inv = np.linalg.pinv(Sigma)

    diff = X_query - mu
    d2 = np.einsum('ij,jk,ik->i', diff, Sigma_inv, diff)
    # Numerical safety: tiny negatives to zero
    d2 = np.maximum(d2, 0.0)
    d = np.sqrt(d2)
    return d, d2, mu, Sigma_inv

# Compute Mahalanobis for candidates vs AFP
maha, maha2, mu_orig, Sigma_inv = mahalanobis_scores(afp_scaled, filtered_scaled, use_shrinkage=True)

#Convert to p-values via chi-square (df = number of features)
df_features = afp_scaled.shape[1]
p_vals = chi2.sf(maha2, df=df_features)  # survival function = 1 - cdf

# Choose a significance threshold (e.g., alpha = 0.01)
alpha = 0.01
is_diverse = p_vals < alpha

# Optional: Control FDR using Benjamini–Hochberg
def benjamini_hochberg(p, q=0.05):
    p = np.asarray(p)
    n = p.size
    order = np.argsort(p)
    ranks = np.arange(1, n+1)
    thresh = (ranks / n) * q
    passed = p[order] <= thresh
    cutoff_idx = np.where(passed)[0].max() if np.any(passed) else None
    mask = np.zeros(n, dtype=bool)
    if cutoff_idx is not None:
        mask[order[:cutoff_idx+1]] = True
    return mask

# Example: use FDR control instead of a fixed alpha
# is_diverse = benjamini_hochberg(p_vals, q=0.05)

# Attach to your candidate DataFrame
filtered_df = E_df.copy()
filtered_df['Mahalanobis'] = maha
filtered_df['Mahalanobis2'] = maha2
filtered_df['p_value'] = p_vals
filtered_df['Diverse_flag'] = is_diverse

# Rank by distance (or by p-value ascending)
ranked = filtered_df.sort_values(['Diverse_flag', 'Mahalanobis'], ascending=[False, False])

# Save results
ranked.to_csv("diverse_candidates_mahalanobis.csv", index=False)
print("Saved: 'diverse_candidates_mahalanobis.csv' (with p-values and flags)")

#3D PCA for visualization (fit on AFP only)
pca = PCA(n_components=3, whiten=True)  # whitened -> Euclidean ≈ Mahalanobis in PC space
afp_pca = pca.fit_transform(afp_scaled)
filtered_pca = pca.transform(filtered_scaled)

# Plot: highlight diverse vs non-diverse
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')

# AFP cloud
ax.scatter(afp_pca[:,0], afp_pca[:,1], afp_pca[:,2],
           c='#1f77b4', alpha=0.25, s=8, label='AFP (reference)')

# Candidates: non-diverse
mask_nd = ~is_diverse
ax.scatter(filtered_pca[mask_nd,0], filtered_pca[mask_nd,1], filtered_pca[mask_nd,2],
           c="#060606", alpha=0.6, s=18, label='Candidates (in-cluster)')

# Candidates: diverse
ax.scatter(filtered_pca[is_diverse,0], filtered_pca[is_diverse,1], filtered_pca[is_diverse,2],
           c='#d62728', alpha=0.95, s=28, label='Candidates (diverse)')

evr = pca.explained_variance_ratio_ * 100
ax.set_xlabel(f'PC1 ({evr[0]:.1f}%)')
ax.set_ylabel(f'PC2 ({evr[1]:.1f}%)')
ax.set_zlabel(f'PC3 ({evr[2]:.1f}%)')
ax.set_title('Diverse Candidates Relative to AFP (Mahalanobis in Original Space)')
ax.view_init(elev=20, azim=35)
ax.legend(loc='best', ncol=2)
ax.grid(True)
plt.tight_layout()
plt.savefig("pca3_diverse_overlay.png", dpi=300, bbox_inches='tight')
plt.show()

