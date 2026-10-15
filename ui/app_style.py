from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication


def load_stylesheet() -> str:
    """
    Load the application stylesheet (.qss).

    Returns
    -------
    str
        Contents of the stylesheet.
    """

    stylesheet_path = Path(__file__).parent / "styles.qss"

    with stylesheet_path.open("r", encoding="utf-8") as file:
        return file.read()


def apply(app: QApplication) -> None:
    """
    Apply the application-wide style.

    Parameters
    ----------
    app
        QApplication instance.
    """

    app.setStyleSheet(load_stylesheet())