[![Documentation Status](https://readthedocs.org/projects/lacuscore/badge/?version=latest)](https://lacuscore.readthedocs.io/en/latest/?badge=latest)

# Modulable Lacus

Lacus, but as a simple module.

# Installation

```bash
pip install lacuscore
```

# Design

`LacusCore` is the part taking care of enqueuing and capturing URLs or web enabled documents.
It can be used as a module in your own project, see below for the usage

[Lacus](https://github.com/ail-project/lacus) is the webservice that uses `LacusCore`,
and you can use [Pylacus](https://github.com/ail-project/pylacus) to query it.

The `enqueue`, `get_capture_status`, and `get_capture` methods if `LacusCore` and `PyLacus` have
the same parameters which means you can easily use them interchangeably in your project.


For more information regarding the usage of the module and a few examples, please refer to
[the documentation](https://lacuscore.readthedocs.io/en/latest/)
