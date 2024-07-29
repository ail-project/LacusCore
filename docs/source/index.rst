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
2. Trigger the enqueued captures themselves, wait for them to finish.
3. Get the capture result with `get_capture` with the UUID from you got from `enqueue`.


Example
-------

The example below is the minimum viable code to use in order to capture a URL.

* Enqueue

.. code:: python

    from redis import Redis
    from lacuscore import LacusCore

    redis = Redis()  # Connector to a running Redis/Valkey instance
    lacus = LacusCore(redis)
    uuid = lacus.enqueue(url='google.fr')

* Trigger the captures with the highest priority from the queue

.. code:: python

    import asyncio

    from redis import Redis

    from lacuscore import LacusCore

    redis = Redis()  # Connector to a running Redis/Valkey instance
    lacus = LacusCore(redis)

    async def run_captures():
        max_captures_to_consume = 10
        captures = set()
        for capture_task in lacus.consume_queue(max_captures_to_consume):
            captures.add(capture_task)  # adds the task to the set
            capture_task.add_done_callback(captures.discard)  # remove the task from the set when done

        await asyncio.gather(*captures)  # wait for all tasks to complete

    asyncio.run(run_captures())

* Capture Status

.. code:: python

    status = lacus.get_capture_status(uuid)

    # 0 = queued / 1 = done / 2 = ongoing / -1 = Unknown UUID

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
