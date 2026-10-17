# RADAR model card

RADAR is a deep learning architecture for longitudinal diagnosis records. At every
encounter it pools an unordered diagnosis set, updates exponentially
decaying memory channels, and uses twelve monthly queries to estimate AF
and death hazards through 365 days.

## Inputs

- Diagnosis codes recorded through the current encounter
- Time between encounters
- Age at the current encounter
- Binary sex

The reference MIMIC adapter maps ICD-10-CM diagnoses to ICD-9-CM with the CMS
General Equivalence Mappings and removes AF outcome codes both before
and after mapping. It does not use notes, medications, procedures, laboratory
measurements, race, or patient identifiers as clinical model features.
Procedures are used only for postoperative AF and prediction-encounter
exclusions, from the surgery date through day 90 afterward by default.

## Outputs

RADAR returns monthly AF cumulative incidence, death cumulative
incidence, and event-free survival. These probabilities are internally
coherent and sum to one at each endpoint.

## Use

The repository provides model training and inference for research, synthetic
examples, and an adapter for locally accessed MIMIC-IV data. The MIMIC workflow
fits a model using the local training partition. The paper's external validation
uses the frozen HCAI-trained model and its vocabulary and normalization.
