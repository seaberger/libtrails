"""Tests for CLI helper functions."""

from unittest.mock import MagicMock, patch

from libtrails.cli import _progress_tracker


class TestProgressTrackerTTY:
    """Tests for _progress_tracker when stdout is a TTY (Rich Progress path)."""

    @patch("libtrails.cli.Progress")
    @patch("libtrails.cli.console")
    def test_yields_callable(self, mock_console, mock_progress_cls):
        """The tracker yields a callable callback."""
        mock_console.is_terminal = True
        mock_progress = MagicMock()
        mock_progress_cls.return_value.__enter__ = MagicMock(return_value=mock_progress)
        mock_progress_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_progress.add_task.return_value = 0

        with _progress_tracker("Testing", 10) as callback:
            assert callable(callback)

    @patch("libtrails.cli.Progress")
    @patch("libtrails.cli.console")
    def test_callback_calls_progress_update(self, mock_console, mock_progress_cls):
        """Callback calls progress.update with correct args."""
        mock_console.is_terminal = True
        mock_progress = MagicMock()
        mock_progress_cls.return_value.__enter__ = MagicMock(return_value=mock_progress)
        mock_progress_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_progress.add_task.return_value = 42

        with _progress_tracker("Testing", 10) as callback:
            callback(5, 10)
            mock_progress.update.assert_called_with(42, completed=5)

            callback(10, 10)
            mock_progress.update.assert_called_with(42, completed=10)


class TestProgressTrackerNonTTY:
    """Tests for _progress_tracker when stdout is NOT a TTY (print fallback)."""

    @patch("libtrails.cli.console")
    def test_yields_callable(self, mock_console):
        """The tracker yields a callable callback in non-TTY mode."""
        mock_console.is_terminal = False
        with _progress_tracker("Testing", 10) as callback:
            assert callable(callback)

    @patch("libtrails.cli.console")
    def test_prints_at_1pct_intervals(self, mock_console):
        """Non-TTY mode prints at every 1% change (threshold: pct >= last_pct + 1)."""
        mock_console.is_terminal = False
        with _progress_tracker("Processing", 100) as callback:
            mock_console.print.reset_mock()

            callback(1, 100)  # 1% — should print (1 >= -1+1=0)
            assert mock_console.print.call_count == 1

            callback(1, 100)  # 1% again — should NOT print (1 < 1+1=2)
            assert mock_console.print.call_count == 1

            callback(2, 100)  # 2% — should print (2 >= 1+1=2)
            assert mock_console.print.call_count == 2

            callback(5, 100)  # 5% — should print (5 >= 2+1=3)
            assert mock_console.print.call_count == 3

    @patch("libtrails.cli.console")
    def test_prints_on_completion(self, mock_console):
        """Non-TTY mode always prints when completed == total."""
        mock_console.is_terminal = False
        with _progress_tracker("Processing", 10) as callback:
            mock_console.print.reset_mock()
            callback(10, 10)
            assert mock_console.print.call_count == 1
            printed = mock_console.print.call_args[0][0]
            assert "10/10" in printed
            assert "100%" in printed

    @patch("libtrails.cli.console")
    def test_handles_zero_total(self, mock_console):
        """Non-TTY mode handles total=0 without division error."""
        mock_console.is_terminal = False
        with _progress_tracker("Processing", 0) as callback:
            callback(0, 0)  # Should not raise

    @patch("libtrails.cli.console")
    def test_description_in_output(self, mock_console):
        """Non-TTY output includes the description string."""
        mock_console.is_terminal = False
        with _progress_tracker("Extracting topics", 10) as callback:
            mock_console.print.reset_mock()
            callback(10, 10)
            printed = mock_console.print.call_args[0][0]
            assert "Extracting topics" in printed
