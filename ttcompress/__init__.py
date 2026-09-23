"""ttcompress — Amortized Utility Attribution for reader-agnostic context
compression (METHOD_SPEC.md).

Pipeline: random chunk masks -> reader answers -> ridge surrogate of the
downstream F1 per chunk (utility attribution, one reader or several) ->
distill the coefficients into a query-aware cross-encoder chunk pruner that
runs in one forward pass -> evaluate at fixed token budgets against the
answer-span oracle and published compressors, per source and per reader.
"""
