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


# Usage

The recommended way to use this module is as follows:

1. Enqueue what you want to capture with `enqueue` (it returns a UUID)
2. Trigger the capture itself. For that, you have two options

  * The `capture` method directly, if you pass it the UUID you got from `enqueue`.
    This is what you want to use to do the capture in the same process as the one enqueuing the capture

  * If you rather want to enqueue the captures in one part of your code and trigger the captures in an other one,
    use `consume_queue` which will pick a capture from the queue and trigger the capture.
    I this case, you should use `get_capture_status` to check if the capture is over before the last step.

3. Get the capture result with `get_capture` with the UUID from you got from `enqueue`.

# Example

## Enqueue

```python

from redis import Redis
from lacuscore import LacusCore

redis = Redis()
lacus = LacusCore(self.redis)
uuid = lacus.enqueue('google.fr')
```

## Capture

* Option 1: Trigger a specific capture via the UUID returned by the `enqueue` call

```python
await lacus.capture(uuid)
```

* Option 2: Trigger the capture with the highest priority from the queue

```python
uuid = await lacus.consume_queue()
```

## Status of a capture

```python
status = lacus.get_capture_status(uuid)
```

## Capture result

```python
result = lacus.get_capture(uuid)
```

