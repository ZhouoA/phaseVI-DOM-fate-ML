# Input data schema

Research data are not distributed in this public code repository. The workflow
accepts either paired raw FT-ICR MS workbooks or an analysis-ready processed
CSV supplied by the user.

Each raw workbook must contain a `result` sheet with these columns:

`formula`, `intens`, `neu.m/z`, `O/C`, `H/C`, `DBE`, `NOSC`, `AI_mod`, `C`,
`H`, `N`, `O`, `P`, `S`, `Cl`, and `Br`.

The script validates non-negative intensities, recalculates relative intensity
within each sample as `intens / sum(intens)`, and requires formula-derived
descriptors, including corrected `NOSC`, to agree for shared formulae.

An analysis-ready CSV must contain formula identifiers, influent and effluent
detection indicators, detection status, `RI_influent`, `RI_effluent`, `FC`,
`DOM_fate`, and all 17 candidate descriptors documented in the main README.
Missing RI or FC values may represent nondetection in one sample and must not
be replaced by zero.

Molecular formulae are analytical observations rather than independent reactor
replicates. Assigned formulae do not confirm molecular structures, relative
intensity is not concentration, and the operational fate labels do not by
themselves establish degradation, production, or persistence mechanisms.
