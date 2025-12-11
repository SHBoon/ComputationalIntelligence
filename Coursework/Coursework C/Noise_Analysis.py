import os
import numpy as np
import scipy.io as spio
from scipy.signal import welch
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt

# -------------------------------------------------
# Config
# -------------------------------------------------
DATA_DIR = "Coursework/Coursework C/Coursework_C_Datasets"
DATASETS = ["D1", "D2", "D3", "D4", "D5", "D6"]
FS = 25000  # Hz, as in your main pipeline

# Length of PSD segments
N_PER_SEG = 4096


def load_dataset(name):
    path = os.path.join(DATA_DIR, f"{name}.mat")
    mat = spio.loadmat(path, squeeze_me=True)
    return np.asarray(mat["d"]).ravel()


def compute_basic_stats(x):
    """Return mean, std, MAD, skew, kurtosis."""
    mean = np.mean(x)
    std = np.std(x)
    mad = np.median(np.abs(x - np.median(x))) / 0.6745  # robust sigma
    sk = skew(x)
    kt = kurtosis(x, fisher=True)  # 0 for Gaussian
    return mean, std, mad, sk, kt


def compute_psd(x, fs=FS, nperseg=N_PER_SEG):
    freqs, psd = welch(x, fs=fs, nperseg=nperseg)
    return freqs, psd


def bandpower(freqs, psd, fmin, fmax):
    """Integrate PSD between fmin and fmax."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    return np.trapz(psd[mask], freqs[mask])


def summarize_noise(name, noise, baseline=None):
    """
    Print stats for 'noise'.
    If 'baseline' (e.g. D1 noise) is provided, also print ratios.
    """
    mean, std, mad, sk, kt = compute_basic_stats(noise)
    freqs, psd = compute_psd(noise)

    # Some example bands (tweak as you like)
    bp_low = bandpower(freqs, psd, 0, 300)        # near-DC / drift
    bp_mid = bandpower(freqs, psd, 300, 3000)     # typical spike band
    bp_high = bandpower(freqs, psd, 3000, 10000)  # higher freq
    bp_total = bandpower(freqs, psd, 0, freqs[-1])

    print(f"\n=== Noise summary: {name} ===")
    print(f"Length: {len(noise)}")
    print(f"Mean:     {mean:.4e}")
    print(f"Std:      {std:.4e}")
    print(f"MAD:      {mad:.4e}")
    print(f"Skewness: {sk:.4f}")
    print(f"Kurtosis (0 ≈ Gaussian): {kt:.4f}")
    print(f"Bandpower total: {bp_total:.4e}")
    print(f"  0–300 Hz:     {bp_low:.4e}  ({bp_low / bp_total * 100 if bp_total > 0 else 0:.1f}% of total)")
    print(f"  300–3k Hz:    {bp_mid:.4e}  ({bp_mid / bp_total * 100 if bp_total > 0 else 0:.1f}% of total)")
    print(f"  3k–10k Hz:    {bp_high:.4e} ({bp_high / bp_total * 100 if bp_total > 0 else 0:.1f}% of total)")

    if baseline is not None:
        _, std_base, mad_base, _, _ = compute_basic_stats(baseline)
        print(f"Std ratio vs baseline: {std / std_base:.3f}")
        print(f"MAD ratio vs baseline: {mad / mad_base:.3f}")

    return freqs, psd


def plot_psd_comparison(noise_dict):
    """
    noise_dict: {label: noise_array}
    Produces a single figure overlaying PSDs for each label.
    """
    plt.figure(figsize=(10, 6))
    for label, noise in noise_dict.items():
        freqs, psd = compute_psd(noise)
        plt.semilogy(freqs, psd, label=label, alpha=0.8)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title("Noise PSD comparison")
    plt.legend()
    plt.tight_layout()


def plot_histograms(noise_dict, nbins=200):
    """
    Simple histograms to compare distribution shapes.
    """
    plt.figure(figsize=(10, 6))
    for label, noise in noise_dict.items():
        # Normalise by std so shapes are comparable
        z = noise / (np.std(noise) + 1e-12)
        hist, bins = np.histogram(z, bins=nbins, density=True)
        centers = 0.5 * (bins[:-1] + bins[1:])
        plt.plot(centers, hist, label=label, alpha=0.8)
    plt.xlabel("Value (normalised by std)")
    plt.ylabel("Density")
    plt.title("Noise amplitude distributions (normalised)")
    plt.legend()
    plt.tight_layout()


def main():
    # -------------------------------------------------
    # Load all datasets
    # -------------------------------------------------
    data = {}
    for ds in DATASETS:
        print(f"Loading {ds}...")
        data[ds] = load_dataset(ds)

    # Sanity check: all same length?
    lengths = {ds: len(sig) for ds, sig in data.items()}
    print("\nSignal lengths:", lengths)

    # -------------------------------------------------
    # Direct noise estimate: Dk_noise = Dk - D1
    # (assumes same underlying spikes & alignment)
    # -------------------------------------------------
    d1 = data["D1"]
    noise_from_diff = {}

    for ds in DATASETS:
        if ds == "D1":
            # Define D1 "noise" as just the raw trace minus its mean
            noise = d1 - np.mean(d1)
            noise_from_diff[ds] = noise
        else:
            # Added noise estimate: Dk - D1
            if len(data[ds]) != len(d1):
                raise ValueError(f"Length mismatch between D1 and {ds}")
            noise = data[ds] - d1
            noise_from_diff[ds] = noise

    # -------------------------------------------------
    # Print numeric summaries
    # -------------------------------------------------
    baseline_noise = noise_from_diff["D1"]

    for ds, noise in noise_from_diff.items():
        if ds == "D1":
            summarize_noise(ds + " (baseline)", noise)
        else:
            summarize_noise(ds + " (Dk − D1)", noise, baseline=baseline_noise)

    # -------------------------------------------------
    # Plots: PSD and histograms
    # -------------------------------------------------
    # Compare only the "added noise" (D2–D6 − D1)
    added_noise_only = {ds: noise for ds, noise in noise_from_diff.items() if ds != "D1"}

    plot_psd_comparison(added_noise_only)
    plot_histograms(added_noise_only)

    plt.show()


if __name__ == "__main__":
    main()