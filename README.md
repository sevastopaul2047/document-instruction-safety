# Document Instruction Safety

This repository is being developed for the **Lexsi - AI Safety and Alignment Hackathon**.

## Research question

Can a language model use helpful information from an external document while ignoring instructions in that document that try to redirect its behavior?

## Approach

We plan to evaluate a Cohere Labs open-weight language model on document-based question-answering tasks. The evaluation will compare normal documents, documents containing injected instructions, and a simple defensive condition.

The pipeline will use **CuratorKIT** for dataset preparation and **AuditKIT** for safety evaluation. We plan to measure attack success, legitimate task accuracy, and unnecessary refusals.

## Status

Initial project setup. Experiments and results have not been completed yet. The dataset, code, evaluation settings, and findings will be documented as the project develops.
