# Poroelastic simulation and equation discovery

Code and data for modeling ensemble-averaged pore pressure and displacement across 443 permeability realizations.

With Python 3.11, run these commands from the project folder:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open a notebook and select **Run All**.

1. [Brownian material fields](01_brownian_material.ipynb) — Generate one permeability field and its material properties.
2. [Single realization](02_single_realization.ipynb) — Run after notebook 01 to simulate and plot pressure and displacement.
3. [Kernel closure](03_kernel_closure.ipynb) — Fit the closure and run the forward model using the included ensemble average.
4. [Fractional PDE](04_fractional_pde.ipynb) — View saved results by default, or set `RUN_MODE = "frozen"` to run the published model.

Sources are cited in each notebook. The [FPDE license](src/fpde/LICENSE) is included. Redistribution permission for the adapted fBm generator is unconfirmed.

Guodong Ren · University of Southern California · guodongr@usc.edu
