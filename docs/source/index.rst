Welcome to LacusCore's documentation!
=====================================

This is the core module used by `Lacus <https://github.com/ail-project/lacus>`_
to enqueue, trigger and get the results of a capture (URL or a web enabled document).


Installation
------------

The package is available on PyPi, so you can install it with::

  pip install lacuscore


Usage
-----


The recommended way to use this module is as follows:

1. Enqueue what you want to capture with `enqueue` (it returns a UUID)
2. Trigger the capture itself. For that, you have two options

  * The `capture` method directly, if you pass it the UUID you got from `enqueue`.
    This is what you want to use to do the capture in the same process as the one enqueuing the capture

  * If you rather want to enqueue the captures in one part of your code and trigger the captures in an other one,
    use `consume_queue` which will pick a capture from the queue and trigger the capture.
    I this case, you should use `get_capture_status` to check if the capture is over before the last step.

3. Get the capture result with `get_capture` with the UUID from you got from `enqueue`.


Example
-------

The example below is the minimum viable code to use in order to capture a URL.

* Enqueue

.. code:: python

    from redis import Redis
    from lacuscore import LacusCore

    redis = Redis()
    lacus = LacusCore(redis)
    uuid = lacus.enqueue('google.fr')

* Capture

  * Option 1: Trigger a specific capture via the UUID returned by the `enqueue` call

    .. code:: python

      await lacus.capture(uuid)

  * Option 2: Trigger the capture with the highest priority from the queue

    .. code:: python

      uuid = await lacus.consume_queue()

* Capture Status

.. code:: python

    status = lacus.get_capture_status(uuid)

* Capture result

.. code:: python

    result = lacus.get_capture(uuid)

Library
-------

And for more details on the library:

.. toctree::
   :glob:

   api_reference


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
