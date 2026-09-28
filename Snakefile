# Workflow reproductible : `snakemake -c4`  (equivalent a `python -m spatialrl.cli all`)
# NB : la logique est testee via la CLI ; ce Snakefile n'a pas ete execute dans l'environnement de redaction.
configfile: "config.yaml"
OUT = config["outdir"]

rule all:
    input: f"{OUT}/SpatialRL-POC_onepager.pdf"

rule simulate:
    output: f"{OUT}/sim.npz"
    shell: "PYTHONPATH=src python -m spatialrl.cli simulate"

rule deconv:
    input: f"{OUT}/sim.npz"
    output: f"{OUT}/deconv_est.npz", f"{OUT}/deconv_summary.json"
    shell: "PYTHONPATH=src python -m spatialrl.cli deconv"

rule robustness:
    input: f"{OUT}/sim.npz"
    output: f"{OUT}/robustness.csv"
    shell: "PYTHONPATH=src python -m spatialrl.cli robustness"

rule niche:
    input: f"{OUT}/deconv_est.npz"
    output: f"{OUT}/niche_ring.csv", f"{OUT}/niche_lr.csv", f"{OUT}/niche_arrays.npz", f"{OUT}/niche_summary.json"
    shell: "PYTHONPATH=src python -m spatialrl.cli niche"

rule coregister:
    input: f"{OUT}/deconv_est.npz"
    output: f"{OUT}/msi_summary.csv", f"{OUT}/msi_registration.json"
    shell: "PYTHONPATH=src python -m spatialrl.cli coregister"

rule report:
    input: f"{OUT}/robustness.csv", f"{OUT}/niche_ring.csv", f"{OUT}/msi_summary.csv"
    output: f"{OUT}/SpatialRL-POC_onepager.pdf", f"{OUT}/report.md", f"{OUT}/results.json"
    shell: "PYTHONPATH=src python -m spatialrl.cli report"
