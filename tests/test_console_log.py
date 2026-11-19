from io import StringIO
import sys
from unittest.mock import Mock

import pytest

from shared.console_log import run_logged, terminal_only


def test_mirrors_both_streams_and_flushes_partial_lines(tmp_path, capsys):
    path = tmp_path / "nested" / "collection.log"
    stdout, stderr, stdin = sys.stdout, sys.stderr, sys.stdin

    def action():
        assert sys.stdin is stdin
        print("partial progress", end="")
        assert "partial progress" in path.read_text(encoding="utf-8")
        assert "partial progress" in capsys.readouterr().out
        print("retry warning", file=sys.stderr)
        return 0

    assert run_logged(path, action) == 0
    assert sys.stdout is stdout
    assert sys.stderr is stderr
    log = path.read_text(encoding="utf-8")
    assert "partial progress" in log
    assert "retry warning" in log
    assert "exit code 0" in log
    assert "retry warning" in capsys.readouterr().err
    path.rename(path.with_suffix(".closed"))


def test_appends_resumed_runs_and_preserves_failure_status(tmp_path):
    path = tmp_path / "collection.log"
    assert run_logged(path, lambda: 0) == 0
    assert run_logged(path, lambda: 1) == 1
    log = path.read_text(encoding="utf-8")
    assert log.count("Run started") == 2
    assert "exit code 0" in log
    assert "exit code 1" in log


def test_interactive_prompts_and_echoed_credentials_are_terminal_only(tmp_path, capsys):
    path = tmp_path / "collection.log"

    def action():
        with terminal_only():
            print("Email: rider@example.test", end="")
            print("secret-password", file=sys.stderr)
        print("Authenticated")
        return 0

    assert run_logged(path, action) == 0
    captured = capsys.readouterr()
    assert "Email: rider@example.test" in captured.out
    assert "secret-password" in captured.err
    log = path.read_text(encoding="utf-8")
    assert "rider@example.test" not in log
    assert "secret-password" not in log
    assert "Authenticated" in log


def test_preserves_terminal_capabilities(monkeypatch, tmp_path):
    terminal = Mock(wraps=StringIO())
    terminal.write.side_effect = lambda text: len(text)
    terminal.isatty.return_value = True
    terminal.fileno.return_value = 123
    terminal.encoding = "utf-8"
    terminal.errors = "strict"
    monkeypatch.setattr(sys, "stdout", terminal)

    def action():
        assert sys.stdout.isatty() is True
        assert sys.stdout.fileno() == 123
        assert sys.stdout.encoding == "utf-8"
        assert sys.stdout.errors == "strict"
        return 0

    assert run_logged(tmp_path / "collection.log", action) == 0


def test_unexpected_error_is_recorded_and_propagated(tmp_path):
    path = tmp_path / "collection.log"
    stdout, stderr = sys.stdout, sys.stderr

    def action():
        raise RuntimeError("unexpected failure")

    with pytest.raises(RuntimeError, match="unexpected failure"):
        run_logged(path, action)
    assert sys.stdout is stdout
    assert sys.stderr is stderr
    log = path.read_text(encoding="utf-8")
    assert "Traceback" in log
    assert "RuntimeError: unexpected failure" in log


def test_keyboard_interrupt_is_recorded(tmp_path):
    path = tmp_path / "collection.log"

    def action():
        raise KeyboardInterrupt

    assert run_logged(path, action) == 130
    assert "Interrupted by user" in path.read_text(encoding="utf-8")


def test_unwritable_log_stops_before_action(tmp_path, capsys):
    path = tmp_path / "is_a_directory"
    path.mkdir()
    action = Mock()
    assert run_logged(path, action) == 1
    action.assert_not_called()
    assert "Cannot open console log" in capsys.readouterr().err
