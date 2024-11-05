#!/usr/bin/env python3

from __future__ import annotations

from typing import Any, TypeVar
from collections.abc import Coroutine

import asyncio
import functools

from .helpers import LacusCoreLogAdapter

T = TypeVar('T')

# Code from https://quantlane.com/blog/ensure-asyncio-task-exceptions-get-logged/


def create_task(
    coroutine: Coroutine[Any, Any, T],
    *,
    name: str,
    logger: LacusCoreLogAdapter,
    message: str,
    message_args: tuple[Any, ...] = (),
    loop: asyncio.AbstractEventLoop | None = None,

) -> asyncio.Task[T]:  # This type annotation has to be quoted for Python < 3.9, see https://www.python.org/dev/peps/pep-0585/
    '''
    This helper function wraps a ``loop.create_task(coroutine())`` call and ensures there is
    an exception handler added to the resulting task. If the task raises an exception it is logged
    using the provided ``logger``, with additional context provided by ``message`` and optionally
    ``message_args``.
    '''
    if loop is None:
        loop = asyncio.get_running_loop()
    task = loop.create_task(coroutine, name=name)
    task.add_done_callback(
        functools.partial(_handle_task_result, logger=logger, message=message, message_args=message_args)
    )
    return task


def _handle_task_result(
    task: asyncio.Task[Any],
    *,
    logger: LacusCoreLogAdapter,
    message: str,
    message_args: tuple[Any, ...] = (),
) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        pass  # Task cancellation should not be logged as an error.
    except asyncio.TimeoutError:
        pass  # Timeout is also fine
    # Ad the pylint ignore: we want to handle all exceptions here so that the result of the task
    # is properly logged. There is no point re-raising the exception in this callback.
    except Exception:  # pylint: disable=broad-except
        logger.exception(message, *message_args)
