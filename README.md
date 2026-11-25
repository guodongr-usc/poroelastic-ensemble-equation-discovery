# Poroelastic simulation and equation discovery for a 443-member ensemble average

Copyright Guodong Ren  
University of Southern California  
Contact: guodongr@usc.edu

## Run the notebooks

| Notebook | Inputs and calculation |
|---|---|
| [01 · Brownian permeability and material fields](01_brownian_material.ipynb) | Generates material realization 027 (seed 68), applies the original constitutive relations and saves a material checkpoint. |
| [02 · One realization](02_single_realization.ipynb) | Run 01 first. Uses its material fields for well case 001, saves 49 simulated frames and plots pressure and displacement at ten times. |
| [03 · Kernel closure](03_kernel_closure.ipynb) | Starts from the bundled EA. Fits the Solution on frames 0–38, solves for ten closure coefficients per equation with the fixed Biot skeleton and gradient amplitude 0.1, and advances the frozen system from the initial condition to day 730. The figure shows all 30 coefficient fields for the intercept, the three field values and their six spatial derivatives. |
| [04 · Fractional PDE](04_fractional_pde.ipynb) | Runs the two-resolution candidate search, selects the equation, advances it from the observed initial fields and evaluates the forecast. The default mode displays the saved result. |

Notebook 03 runs independently of notebooks 01 and 02. In a Python 3.11 environment with `requirements.txt` installed, open notebook 03 and select **Run All**. For a terminal session or an allocated CPU compute node, the equivalent command from the extracted directory is

```bash
python - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient

book = nbformat.read("03_kernel_closure.ipynb", as_version=4)
NotebookClient(book, kernel_name="python3", timeout=None).execute()
nbformat.write(book, Path("outputs") / "03_kernel_closure.executed.ipynb")
PY
```

The coefficient maps and numerical results are written to `outputs/kernel/`.

In notebook 04, set `RUN_MODE` in the first code cell and select **Run All**.

| Mode | Calculation |
|---|---|
| `"saved"` | Scores the supplied Round 4 prediction against the EA and draws the figures. |
| `"frozen"` | Solves the supplied fractional equation from the observed initial fields and scores the resulting prediction. |
| `"local"` | Builds a classical Biot starting equation from the EA material properties and training scales, then runs ten search trials. Each trial uses a population of 64, with 30 coarse-grid and 6 fine-grid generations. The selected equation supplies the parameters for the subsequent forward solve and scoring. |
| `"carc"` | Submits the same search and forward solve as dependent Slurm jobs. Set the partition and account for your allocation. After the jobs finish, run the notebook again in the same directory to view that run's scores and figures. |

The published metrics below refer to the supplied final model. The independent search uses a different initialization from the historical fit and can select a different equation.

## Files and results

- `data/` holds the original EA, its metadata and the frame split. There are 443 retained material members, one well, 49 frames and 10,201 nodes.
- `src/` contains the numerical code called by the notebooks and the optional discovery and frozen-forward routines.
- `assets/` holds the frozen model parameters, solver inputs and the saved FPDE prediction. Notebook 03 generates the closure coefficient figure from the coefficients it fits during execution.
- `provenance/` records the original simulation settings, the kernel closure configuration and the selected FPDE trial.

The later-time evaluation pools frames 39–48 over each field's free nodes. The kernel closure gives R² of 0.991945, 0.995521 and 0.982535 for pressure, horizontal displacement and vertical displacement, respectively. The Round 4 FPDE gives 0.939260, 0.965699 and 0.906044. The later interval was examined during model development.

The FPDE scores describe a forward solve from observed initial displacements. The selected candidate's own equilibrium displacements differed from those observations by 8.08% horizontally and 24.39% vertically. The original per-component compatibility threshold was 5%.

## Source attribution

The Guodong Ren notice covers the notebook preparation, integration and changes for this release. The first code cell of each notebook lists the relevant sources. Implementation files identify adapted routines and the publications describing their methods.

- **01 · Material fields.** The fractional Brownian sampler and its scale calculation adapt four routines from Binhao Li's `FlowSimulation`. The associated study describes the sampler and credits Stein's construction. The material conversion uses a normalized Kozeny–Carman relation and adapts the elastic-modulus law of Osorio and coauthors to two dimensions. [Generator source at the version used for comparison](https://github.com/lastoor/FlowSimulation/blob/4afc232100a16e8c6e636a534da732c5d126f947/simulation/grf_generation.py) · [Li, de Barros & Sahimi, 2026](https://doi.org/10.1029/2026GL124085) · [Stein, 2002](https://doi.org/10.1198/106186002466) · [Carman, 1937, reprinted 1997](https://doi.org/10.1016/S0263-8762(97)80003-2) · [Osorio et al., 2025, Eq. 3](https://doi.org/10.1029/2024JB030836)
- **02 · Single realization.** The coupled pressure and displacement equations use Biot's consolidation theory. The two-dimensional finite-element implementation, well configuration and time schedule are specific to this study. [Biot, 1941](https://doi.org/10.1063/1.1712886)
- **03 · Kernel closure.** The two-step strategy for estimating a solution and then an equation follows the framework of Jalalian and coauthors. The fixed Biot operator, discrete solution basis and ten-term nodal closure are implementation choices for this EA. [Jalalian et al., 2025, §2.1, version 2](https://arxiv.org/abs/2503.01036v2)
- **04 · Fractional PDE.** The search draws on Im and coauthors' work on symbolic regression and stochastic optimization. This implementation searches a poroelastic candidate library. Its spatial stencils come from the archived B-RefStruct solver. Their source notice and license are retained in the package. [Im et al., 2023](https://doi.org/10.1103/PhysRevE.107.L013301) · [Jinwoo Im's reference implementation](https://github.com/Jinwoousc/GPSI) · [Stencil attribution](src/fpde/spatial_kernels.py) · [FPDE license](src/fpde/LICENSE)

The historical CARC generator archive and the inspected `FlowSimulation` revision provide no explicit redistribution license for the fBm implementation. Attribution records its origin. Permission to redistribute that adaptation still needs confirmation from the rights holder. No new license is granted here for third-party components or for the dataset.

This package includes code for fitting the kernel closure and searching for FPDE candidates. Redistribution includes those implementations.
