"""Figures from complete exported samples, with explicit population labels."""
import csv
import json
import os
from pathlib import Path

import numpy as np
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".tmp/v1_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import write_json


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def make_plots(output):
    output = Path(output)
    folder = output / "plots"
    folder.mkdir(exist_ok=True)
    geometry = [r for r in read_rows(output / "geometry_blocks.csv") if r["representation"] == "reconstructed"]
    candidates = read_rows(output / "candidate_tightness.csv")
    certificates = read_rows(output / "certificate_cases.csv")
    layers = sorted({int(r["layer_index"]) for r in geometry})
    definitions = []
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})

    def save(name, fig, population):
        fig.tight_layout()
        fig.savefig(folder / (name + ".pdf"))
        fig.savefig(folder / (name + ".png"), dpi=160)
        plt.close(fig)
        definitions.append({"plot": name, "population": population, "all_samples_in_population_used": True})

    fig, ax = plt.subplots(figsize=(6, 4))
    for layer in layers:
        spectra = [np.asarray(json.loads(r["singular_values"])) for r in geometry if int(r["layer_index"]) == layer]
        width = max(map(len, spectra))
        padded = np.array([np.pad(s, (0, width - len(s))) for s in spectra])
        ax.plot(range(1, width + 1), padded.mean(axis=0), marker="o", label=f"Layer {layer} (n={len(spectra)})")
    ax.set(xlabel="Centered singular-value index", ylabel="Mean singular value (key units)")
    ax.legend(fontsize=8)
    save("01_singular_spectrum", fig, "All unique reconstructed KV-head blocks; short spectra zero-padded")
    for filename, field, label in [("02_effective_rank", "effective_rank_energy_entropy", "Effective rank (energy entropy)"),
                                   ("03_block_radius", "radius_first_anchor", "Radius about first anchor (key units)"),
                                   ("04_anisotropy", "anisotropy_sigma1_over_mean", "Largest / mean singular value")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        values = [[float(r[field]) for r in geometry if int(r["layer_index"]) == layer] for layer in layers]
        ax.boxplot(values, tick_labels=[f"{l}\nn={len(v)}" for l, v in zip(layers, values)], showfliers=True)
        ax.set(xlabel="Layer", ylabel=label)
        save(filename, fig, "All unique reconstructed KV-head blocks; all outliers shown")
    for filename, field, label in [("05_logit_slack", "logit_bound_slack", "Logit upper bound minus actual maximum"),
                                   ("06_mass_ratio", "log_mass_bound_ratio", "log(U_m / actual block mass)")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        values = [float(r[field]) for r in candidates]
        ax.hist(values, bins=40, color="0.35", edgecolor="white")
        ax.set(xlabel=label, ylabel=f"Candidate decisions (N={len(values)})")
        save(filename, fig, "All query-head/block decisions; repeated physical blocks are separate query decisions")
    fig, ax = plt.subplots(figsize=(6, 4))
    selected = [float(r["bound_to_error_ratio"]) for r in certificates if int(r["certified_skips"]) > 0]
    if selected:
        ax.hist(selected, bins=min(15, len(selected)), color="0.35", edgecolor="white")
    else:
        ax.text(.5, .5, "No certified skips in this sample", ha="center", transform=ax.transAxes)
    ax.set(xlabel="Joint-set certificate / measured joint-set error", ylabel=f"Certified cases (N={len(selected)})")
    save("07_certificate_tightness", fig, "Certified query cases only; one joint-set ratio per case, zero-skip cases excluded")
    fig, ax = plt.subplots(figsize=(6, 4))
    skipped = sum(r["certified"] == "True" for r in candidates)
    ax.bar(["Certified skip", "Retained"], [skipped, len(candidates) - skipped], color=["0.25", "0.65"])
    ax.set(ylabel="Query-head/block decisions")
    for x, n in enumerate([skipped, len(candidates) - skipped]):
        ax.text(x, n, str(n), ha="center", va="bottom")
    ax.margins(y=.15)
    save("08_certified_vs_retained", fig, "All query-head/block decisions; linear counts, no physical-skip interpretation")
    ranks = {r["geometry_id"]: float(r["effective_rank_energy_entropy"]) for r in geometry}
    fig, ax = plt.subplots(figsize=(6, 4))
    for state, marker, color in [("False", ".", "0.6"), ("True", "x", "0.05")]:
        items = [r for r in candidates if r["certified"] == state]
        ax.scatter([ranks[r["geometry_id"]] for r in items], [float(r["log_mass_bound_ratio"]) for r in items],
                   marker=marker, s=18, alpha=.6, color=color, label=f"{'Skipped' if state == 'True' else 'Retained'} (n={len(items)})")
    ax.set(xlabel="Block effective rank", ylabel="log(U_m / actual block mass)")
    ax.legend()
    save("09_geometry_vs_tightness", fig, "All query-head/block decisions joined to reconstructed geometry; association only")
    write_json(folder / "plot_populations.json", definitions)


if __name__ == "__main__":
    import sys
    make_plots(Path(sys.argv[1]))
