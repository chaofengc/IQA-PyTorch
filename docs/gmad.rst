Candidate-pool gMAD-style testing
=================================

The standalone ``pyiqa-gmad`` command searches a supplied image candidate pool
for pairs where one no-reference (NR) metric assigns similar scores while
another metric assigns scores as far apart as possible. It uses pyiqa models
for inference by default, or can read scores computed by another tool.

This is a **candidate-pair exploration helper**, not a complete implementation
of the official group maximum differentiation (gMAD) evaluation protocol. It
does not select or constrain a reference population, collect human judgments,
or establish that either metric is correct.

Install
-------

Install pyiqa from PyPI or use an editable checkout:

.. code-block:: bash

   pip install pyiqa

The ``pyiqa-gmad`` entry point is included in current package builds. When
working from a source checkout, run ``pip install -e .``.

Score images with pyiqa
-----------------------

Provide a directory containing at least two images. The directory is searched
recursively; the default metrics are ``musiq`` and ``brisque``.

.. code-block:: bash

   pyiqa-gmad ./candidate_images \
     --metric-a musiq \
     --metric-b brisque \
     --tie-tolerance 0.1 \
     --device cuda \
     --output gmad_results.json

Use precomputed scores
----------------------

The CSV must contain an ``image`` column and one column for each selected
metric. Each row must refer to an existing image. Relative paths are resolved
from the CSV directory unless ``--image-root`` is supplied.

.. code-block:: text

   image,musiq,brisque
   images/sample1.png,62.4,28.1
   images/sample2.png,48.7,39.5

.. code-block:: bash

   pyiqa-gmad --scores-csv scores.csv \
     --metric-a musiq \
     --metric-b brisque \
     --tie-tolerance 0.1

For custom metric names that are not in pyiqa's model configuration, specify
whether higher or lower scores indicate better quality:

.. code-block:: bash

   pyiqa-gmad --scores-csv scores.csv \
     --metric-a model_x --metric-b model_y \
     --direction-a higher --direction-b lower

Interpretation and options
--------------------------

Before pair selection, each metric's scores are standardized separately within
the supplied candidate pool. Scores for metrics marked ``lower_better`` are
sign-reversed so that higher standardized values consistently indicate better
quality. The ``--tie-tolerance`` value is the maximum absolute standardized
score difference allowed for the metric treated as tied; the default is
``0.1``. This is a configurable search tolerance, not a universal gMAD
threshold. Since standardization depends on the candidate pool, changing the
pool can change both standardized scores and selected pairs.

The JSON output reports one pair for each direction:

* metric A ties while metric B differs as much as possible;
* metric B ties while metric A differs as much as possible.

If no pair satisfies a direction's tie constraint, that result is ``null``.
Raw and standardized scores are included for inspection. ``--output`` writes
JSON to a file; without it, results are printed to standard output.

Run ``pyiqa-gmad --help`` for all command-line options.
