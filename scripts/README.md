# scripts

Entry-point scripts for common workflows: training (`train.py`), evaluation
(`evaluate.py`), data preparation, and model export. Scripts are thin wrappers
around the `oilspill` package — they parse arguments, load a config from
`configs/`, and call into library code.
