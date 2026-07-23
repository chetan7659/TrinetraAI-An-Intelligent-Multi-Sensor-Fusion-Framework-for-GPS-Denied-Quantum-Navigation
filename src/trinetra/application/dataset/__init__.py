"""Application-layer dataset services.

This package contains application services that orchestrate the data
ingestion pipeline components defined in:

* ``trinetra.adapters.datasets`` (adapters)
* ``trinetra.domain.interfaces`` (domain contracts)

Services in this package coordinate collaborators — they do NOT perform
parsing, decoding, mapping, preprocessing, or feature engineering.
"""
