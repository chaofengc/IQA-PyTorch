Performance Benchmark and Calibration
==========================================

.. raw:: html

    <style>
    table {
        width: 100%;
        table-layout: auto;
        overflow-x: auto;
    }
    </style>


Performance Evaluation Protocol
----------------------------------------

**We use official models for evaluation if available.** Otherwise, we use the following settings to train and evaluate different models for simplicity and consistency:

.. csv-table:: Evaluation Protocol 
   :header: "Metric Type", "Train", "Test"

    "FR", "KADID-10k", "CSIQ, LIVE, TID2008, TID2013"
    "NR", "KonIQ-10k", "LIVEC, KonIQ-10k (official split), TID2013, SPAQ"
    "Aesthetic IQA", "AVA", "AVA (official split)"

Results are calculated with:

- **PLCC without any correction.** Although test time value correction is common in IQA papers, we want to use the original value in our benchmark.
- **Full image single input.** We use multi-patch testing only when it is necessary for the model to work.

Basically, we use the largest existing datasets for training, and cross dataset evaluation performance for fair comparison. The following models do not provide official weights, and are retrained by our scripts:

- NR: ``cnniqa``, ``dbcnn``, ``hyperiqa``
- Aesthetic IQA: ``nima``, ``nima-vgg16-ava``

Performance on FR benchmarks
----------------------------------------

.. csv-table:: FR benchmark
    :header-rows: 1
    :file: ../tests/FR_benchmark_results.csv

Performance on NR benchmarks
----------------------------------------

.. csv-table:: NR benchmark
    :header-rows: 1
    :file: ../tests/NR_benchmark_results.csv

Performance on image aesthetic benchmarks
----------------------------------------

.. csv-table:: IAA benchmark
    :header-rows: 1
    :file: ../tests/IAA_benchmark_results.csv


Results Calibration
----------------------------------------

.. csv-table:: Calibration
    :header-rows: 1
    :file: ../ResultsCalibra/calibration_summary.csv