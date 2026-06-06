"""Shared control-flow helpers."""

import time

class StopDrawingException(Exception):
    pass


class ResponsiveYield:
    """Small cooperative yield helper for CPU-heavy Python loops.

    The drawing app already runs heavy preparation in a background thread,
    but pure-Python loops can still hold the GIL long enough to make Qt feel
    frozen.  Calling maybe() periodically gives the UI thread a chance to
    repaint timers, buttons, logs, and the computing animation.
    """

    def __init__(self, interval=0.015):
        self.interval = float(interval)
        self.last = time.perf_counter()

    def maybe(self):
        now = time.perf_counter()
        if now - self.last >= self.interval:
            time.sleep(0)
            self.last = time.perf_counter()


def ui_yield():
    time.sleep(0)


def raise_if_stopped(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise StopDrawingException()
