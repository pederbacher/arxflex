# arxflex

ARX models with nonlinear input blocks, specified as formulas: B-splines over
time of day, sigmoid and PCHIP valve curves. The nonlinear parameters are
found by optimisation, the linear ARX parameters by least squares for each
trial, and the fitted model can be simulated multi-step.

The methods are described in

> C. A. Thilker, P. Bacher, D. Cali and H. Madsen (2022). Identification of
> non-linear autoregressive models with exogenous inputs for room air
> temperature modelling. *Energy and AI*, 9, 100165.
> <https://doi.org/10.1016/j.egyai.2022.100165>

```python
from arxflex import fitmodel, print_fit

fit = fitmodel(df, "Ti", {
    "AR": 2,
    "ambient": "Ta - Ti",
    "solar": "s(timeofday, k=5) * Ps",
    "heating": "v(sigmoid, Tset - Ti) * (Tfor - Ti)",
})
print_fit(fit)
```

For a worked example, see the vignette, rendered with all figures at
<https://pederbacher.github.io/arxflex/>; its source is
[`examples/borgerskolen/vignette.py`](https://github.com/pederbacher/arxflex/blob/main/examples/borgerskolen/vignette.py). It
loads the example data, fits and compares models, and analyses the results.
Run it cell by cell, or render it to HTML with
`quarto render examples/borgerskolen/vignette.py`.

## Install

```sh
pip install arxflex                    # the core: numpy and scipy only
pip install "arxflex[pandas,plot]"     # with data-frame input and plotting
pip install "arxflex[examples]"        # also the example data and what the vignette needs
```

Directly from GitHub:

```sh
pip install "arxflex[pandas,plot] @ git+https://github.com/pederbacher/arxflex"
pip install "arxflex-borgerskolen @ git+https://github.com/pederbacher/arxflex#subdirectory=examples/borgerskolen"
```

For development, from a clone: `pip install -e ".[pandas,plot]" -e examples/borgerskolen`.

## Repository layout

```
src/arxflex/                  the package
examples/borgerskolen/        example: room temperatures in Borgerskolen, March 2021
  src/arxflex_borgerskolen/   installed with the example: processed data + helpers
  legacy/                     the original notebook and the hand-written models
  check_fitmodel.py           checks that fitmodel() reproduces the legacy models
  vignette.py                 worked example: fit, compare and analyse models
```

Only `src/arxflex` is installed by the package, and only
`examples/borgerskolen/src/arxflex_borgerskolen` (with the 1.6 MB processed
data file) by the example. The legacy code lives only in the repository.

## Citing

If you use arxflex in your work, please cite the article above:

```bibtex
@article{thilker2022identification,
  title     = {Identification of non-linear autoregressive models with exogenous
               inputs for room air temperature modelling},
  author    = {Thilker, Christian Ankerstjerne and Bacher, Peder and Cali, Davide
               and Madsen, Henrik},
  journal   = {Energy and AI},
  volume    = {9},
  pages     = {100165},
  year      = {2022},
  publisher = {Elsevier},
  doi       = {10.1016/j.egyai.2022.100165}
}
```

## Licence

arxflex is free software under the GNU General Public License, version 3 or
later (GPL-3.0-or-later); see [`LICENSE`](https://github.com/pederbacher/arxflex/blob/main/LICENSE). The example data are under
CC BY 4.0; see [`examples/borgerskolen/DATA_LICENSE.md`](https://github.com/pederbacher/arxflex/blob/main/examples/borgerskolen/DATA_LICENSE.md).
