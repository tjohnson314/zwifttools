"""Mirror Python console output to an append-only log without redirecting stdin."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import sys
import traceback
from typing import Callable, Iterator, TextIO


class _Tee:
    def __init__(self, terminal: TextIO, log: TextIO):
        self.terminal = terminal
        self.log = log

    def write(self, text: str) -> int:
        self.log.write(text)
        self.log.flush()
        written = self.terminal.write(text)
        self.terminal.flush()
        return written

    def flush(self) -> None:
        self.log.flush()
        self.terminal.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    def fileno(self) -> int:
        return self.terminal.fileno()

    @property
    def encoding(self) -> str:
        return self.terminal.encoding

    @property
    def errors(self) -> str:
        return self.terminal.errors


@contextmanager
def terminal_only() -> Iterator[None]:
    """Keep interactive prompts and any echoed input out of the log."""
    stdout = sys.stdout.terminal if isinstance(sys.stdout, _Tee) else sys.stdout
    stderr = sys.stderr.terminal if isinstance(sys.stderr, _Tee) else sys.stderr
    with redirect_stdout(stdout), redirect_stderr(stderr):
        yield


def run_logged(path: Path, action: Callable[[], int]) -> int:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        log = path.open("a", encoding="utf-8")
    except OSError as exc:
        print(f"Cannot open console log {path}: {exc}", file=sys.stderr)
        return 1
    with log, redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
        print(
            f"\nRun started {datetime.now(timezone.utc).isoformat()}; log: {path}",
            file=sys.stderr,
        )
        try:
            result = action()
        except KeyboardInterrupt:
            print("Interrupted by user; completed caches are preserved.", file=sys.stderr)
            result = 130
        except Exception:
            # Record unexpected failures before the console streams are restored.
            traceback.print_exc(file=sys.stderr)
            raise
        print(
            f"Run finished {datetime.now(timezone.utc).isoformat()}; exit code {result}.",
            file=sys.stderr,
        )
        return result
