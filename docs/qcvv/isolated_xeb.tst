# Replacements to apply to notebook when testing the notebook.
# See devtools/notebook_test.py for syntax.

n_library_circuits=20->n_library_circuits=1
repetitions=10_000->repetitions=10
max_depth = 100->max_depth = 10
fatol=5e-3->fatol=100
xatol=5e-3->xatol=100

