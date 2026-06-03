"""§V experiment grid for slack-certify-mapf.

The :mod:`experiments.runners` subpackage holds one
:class:`ExperimentRunner` subclass per research question; the YAML
configs under :mod:`experiments.configs` parametrise their sweep
grids. ``python -m experiments`` dispatches to the right runner
based on the config's ``experiment_name`` field — see
:mod:`experiments.__main__`.
"""
