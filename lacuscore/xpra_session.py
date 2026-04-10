#!/usr/bin/env python3

from __future__ import annotations

import sys

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import select
import subprocess
from tempfile import gettempdir
import time
from typing import Any


from .session import Session

if sys.version_info >= (3, 11):
    from datetime import UTC


logger = logging.getLogger(__name__)


@dataclass
class XpraSession(Session):
    """Representation of a single xpra-backed interactive session.

    This extends the backend-agnostic :class:`Session` with the X11
    display identifier and unix socket transport details used by xpra.
    The common lifetime metadata is shared across any future session
    backends.
    """

    display: str
    socket_path: str


class XpraSessionManager:
    """Manage xpra-based interactive browser sessions over per-session unix sockets.

    Each interactive session starts its own xpra server bound to a local unix
    socket with HTML5 enabled. This keeps the session transport private to the
    Lacus deployment while allowing a separate reverse proxy or sidecar to
    expose a stable end-user route.
    """

    backend_type = 'xpra'
    _BASE_ENV_KEYS = (
        'HOME',
        'LANG',
        'LC_ALL',
        'LC_CTYPE',
        'LOGNAME',
        'PATH',
        'TMPDIR',
        'TZ',
        'USER',
        'XAUTHORITY',
        'XDG_RUNTIME_DIR',
    )

    def __init__(self, xpra_command: str='xpra',
                 socket_dir: str | Path | None=None) -> None:
        """Initialize an xpra session manager.

        ``xpra_command`` defaults to the ``xpra`` binary on PATH and can be
        overridden by passing an explicit path. ``socket_dir`` defaults to a
        private runtime directory and can be overridden explicitly.
        """
        self.xpra_command = xpra_command

        if socket_dir is None:
            runtime_dir = os.getenv('XDG_RUNTIME_DIR')
            if runtime_dir:
                socket_dir = Path(runtime_dir) / 'lacus-xpra'
            else:
                socket_dir = Path(gettempdir()) / 'lacus-xpra'
        self.socket_dir = Path(socket_dir)
        self.socket_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _get_socket_path(self, session_name: str) -> Path:
        return self.socket_dir / f'{session_name}.sock'

    def _build_clean_env(self, extra_env: Mapping[str, str] | None=None) -> dict[str, str]:
        env = {
            'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
            'HOME': os.environ.get('HOME', str(Path.home())),
            'LANG': os.environ.get('LANG', 'C.UTF-8'),
        }

        for key in self._BASE_ENV_KEYS:
            value = os.environ.get(key)
            if value:
                env[key] = value

        if extra_env:
            env.update(extra_env)

        return env

    def _cleanup_socket_path(self, socket_path: str) -> None:
        path = Path(socket_path)
        if not path.exists():
            return

        try:
            path.unlink()
        except FileNotFoundError:
            return
        except Exception as e:
            logger.warning('Unable to remove xpra socket %s: %s', path, e)

    def _wait_for_socket_removal(self, socket_path: str, *, timeout: float=10.0) -> bool:
        path = Path(socket_path)
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if not path.exists():
                return True
            time.sleep(0.25)

        return not path.exists()

    def _wait_for_socket_creation(self, socket_path: str, *, timeout: float=10.0) -> bool:
        path = Path(socket_path)
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if path.exists():
                return True
            time.sleep(0.1)

        return path.exists()

    def _run_stop_command(self, target: str) -> subprocess.CompletedProcess[str] | None:
        cmd = [self.xpra_command, 'stop', target]

        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
                env=self._build_clean_env(),
            )
        except FileNotFoundError as e:
            logger.warning('xpra command not found while stopping %s: %s', target, e)
            return None
        except subprocess.TimeoutExpired:
            logger.warning('xpra stop %s timed out.', target)
            return None
        except Exception as e:
            logger.warning('xpra stop %s failed: %s', target, e)
            return None

    def start_session(self, *, session_name: str, ttl: int) -> XpraSession:
        """Start an xpra session with the given name and allocate a display dynamically.

        :param session_name: Unique name for the interactive session (e.g. UUID).
        :param ttl: Time-to-live in seconds for the interactive session.

        :return: XpraSession describing the running xpra process and its
            internal transport details.
        """
        if sys.version_info >= (3, 11):
            created_at = datetime.now(UTC)
        else:
            created_at = datetime.utcnow()

        expires_at = created_at + timedelta(seconds=ttl)

        read_fd, write_fd = os.pipe()
        socket_path = self._get_socket_path(session_name)
        if socket_path.exists():
            socket_path.unlink()

        cmd = [
            self.xpra_command,
            'seamless',
            '--daemon=no',
            '--attach=no',
            '--html=on',
            '--start-new-commands=no',
            '--system-tray=no',
            '--notifications=no',
            '--file-transfer=no',
            '--open-files=no',
            '--open-url=no',
            '--printing=no',
            '--clipboard=no',
            f'--bind={socket_path}',
            f'--session-name={session_name}',
            f'--displayfd={write_fd}',
        ]

        try:
            xpra_proc = subprocess.Popen(
                cmd,
                pass_fds=(write_fd,),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                env=self._build_clean_env(),
            )
        except FileNotFoundError as e:
            os.close(write_fd)
            os.close(read_fd)
            msg = f"xpra command not found: {self.xpra_command} ({e})"
            logger.error(msg)
            raise RuntimeError(msg) from e

        os.close(write_fd)

        # Wait up to 30 s for xpra to write the display number.
        # Without a timeout, a misconfigured xpra would block the thread forever.
        ready, _, _ = select.select([read_fd], [], [], 30)
        if not ready:
            os.close(read_fd)
            raise RuntimeError(
                'xpra did not write a display number within 30 s. '
                'Check that xpra is correctly installed and that all required '
                'system libraries (e.g. python3-gi) are available.'
            )

        # Read the allocated display number from xpra.
        # xpra writes the display number (e.g., "14\n") to the fd.
        display_bytes = os.read(read_fd, 64)
        os.close(read_fd)

        if not display_bytes:
            stderr_output = ""
            if xpra_proc.stderr is not None:
                try:
                    # Give xpra a brief moment to exit and flush stderr.
                    xpra_proc.wait(timeout=5)
                except Exception:
                    # Ignore wait errors; we are already failing startup.
                    pass
                try:
                    stderr_output = xpra_proc.stderr.read() or ""
                except Exception:
                    stderr_output = ""

            msg = "Failed to read allocated display from xpra"
            if stderr_output:
                # Log the full stderr but only attach the tail to the
                # exception message to keep it concise.
                logger.error("xpra failed to start. Command: %s; stderr: %s", cmd, stderr_output)
                last_line = stderr_output.strip().splitlines()[-1]
                msg = f"{msg}: {last_line}"
            else:
                logger.error("xpra failed to start. Command: %s; no stderr output.", cmd)

            raise RuntimeError(msg)

        display_num = display_bytes.decode('utf-8').strip()
        if not display_num:
            logger.error("xpra returned empty display string. Command: %s", cmd)
            raise RuntimeError("Failed to read allocated display from xpra")

        display = f":{display_num}"

        if not self._wait_for_socket_creation(str(socket_path)):
            stderr_output = ""
            if xpra_proc.stderr is not None:
                try:
                    if xpra_proc.poll() is not None:
                        stderr_output = xpra_proc.stderr.read() or ""
                except Exception:
                    stderr_output = ""

            msg = f'xpra did not create its unix socket at {socket_path} within 10 s'
            if stderr_output:
                logger.error("xpra failed to create socket. Command: %s; stderr: %s", cmd, stderr_output)
                last_line = stderr_output.strip().splitlines()[-1]
                msg = f'{msg}: {last_line}'
            else:
                logger.error("xpra did not create socket %s in time. Command: %s", socket_path, cmd)
            raise RuntimeError(msg)

        return XpraSession(
            created_at=created_at,
            expires_at=expires_at,
            display=display,
            socket_path=str(socket_path),
        )

    def serialize_backend_metadata(self, session: Session) -> Mapping[str, Any]:
        if not isinstance(session, XpraSession):
            raise TypeError(f'Expected XpraSession, got {type(session)!r}')
        return {
            'display': session.display,
            'socket_path': session.socket_path,
        }

    def restore_session(self, *, created_at: datetime, expires_at: datetime,
                        backend_metadata: Mapping[str, Any]) -> XpraSession:
        return XpraSession(
            created_at=created_at,
            expires_at=expires_at,
            display=str(backend_metadata.get('display', '')),
            socket_path=str(backend_metadata.get('socket_path', '')),
        )

    def get_capture_env(self, session: Session) -> Mapping[str, str | float | bool]:
        """returns ENV variables to pass to the capture"""
        if not isinstance(session, XpraSession):
            raise TypeError(f'Expected XpraSession, got {type(session)!r}')
        return {'DISPLAY': session.display}

    def stop_session(self, session: Session) -> bool:
        """Terminate a running xpra backend session.

        The implementation delegates to the ``xpra stop`` command,
        targeting the display associated with this session. This keeps
        the shutdown logic in xpra itself and avoids relying on PIDs
        being stable across restarts.

        If the session is already gone or xpra is not reachable, the
        call is treated as a best-effort no-op. The return value signals
        whether shutdown was confirmed or the backend already appeared to
        be gone. This method does not perform any Redis or higher-level
        cleanup; that must be handled by the caller.
        """
        if not isinstance(session, XpraSession):
            raise TypeError(f'Expected XpraSession, got {type(session)!r}')

        socket_target = f'socket://{session.socket_path}'
        stop_targets = [socket_target]
        if session.display:
            stop_targets.append(session.display)

        for target in stop_targets:
            completed = self._run_stop_command(target)
            if completed is None:
                continue

            output = ' '.join(part for part in [completed.stdout, completed.stderr] if part).lower()
            already_stopped = any(marker in output for marker in [
                'no server',
                'not running',
                'not found',
                'cannot find',
                'no display',
            ])
            success = completed.returncode == 0 or already_stopped

            if not success:
                logger.warning('xpra stop %s failed with code %s: %s', target, completed.returncode, output.strip())
                continue

            if already_stopped:
                self._cleanup_socket_path(session.socket_path)
                return True

            if self._wait_for_socket_removal(session.socket_path):
                self._cleanup_socket_path(session.socket_path)
                return True

            logger.warning('xpra stop %s succeeded but socket %s is still present after waiting.',
                           target, session.socket_path)

        return False
