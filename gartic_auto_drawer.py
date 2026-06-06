"""Compatibility entry point for Gartic OpenCV Drawer.

The application code lives in the :mod:`gartic_drawer` package so the project
is easier to browse and maintain, while this file keeps the original
``python gartic_auto_drawer.py`` launch command working.
"""

from gartic_drawer.app import main


if __name__ == "__main__":
    raise SystemExit(main())
