import re
from logging import FileHandler, Formatter, LogRecord
from typing import override
from warnings import warn


class DecoloringFileHandler(FileHandler):
    ansi_escape: re.Pattern[str]

    def __init__(self, filename, mode="a", encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    @override
    def format(self, record: LogRecord) -> str:
        return self.ansi_escape.sub("", super().format(record))


try:
    import colorlog

    class ColoredFormatter(colorlog.ColoredFormatter):
        pass
except ImportError:
    warn("colorlog not installed, colored logging will not be available.")

    class ColoredFormatter(Formatter):
        def __init__(
            self,
            fmt: str | None = None,
            datefmt: str | None = None,
            style="%",
            validate: bool = True,
            *,
            defaults=None,
            **kwargs,
        ) -> None:
            super().__init__(fmt, datefmt, style, validate, defaults=defaults)
