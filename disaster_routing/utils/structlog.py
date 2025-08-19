import json
from os import environ
from typing import override
from warnings import warn

reset_all = ""
bright = ""
dim = ""
white = ""


def colorize(json: str) -> str:
    return json


try:
    from colorama import Fore, Style, just_fix_windows_console
    from pygments import formatters, highlight, lexers

    color_enabled = environ.get("COLOR", "1") in ("1", "true", "yes", "on", "enable")
    if color_enabled:
        just_fix_windows_console()

        reset_all = Style.RESET_ALL
        bright = Style.BRIGHT
        dim = Style.DIM
        white = Fore.WHITE

        def colorize(json: str) -> str:
            return highlight(
                json, lexers.JsonLexer(), formatters.TerminalFormatter(bg="dark")
            ).strip()

except ImportError:
    warn("colorama not installed, color output will not be available.")
    color_enabled = False


def json_default(obj: object) -> object:
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


class StructLog:
    message: str
    kwargs: dict[str, object]

    def __init__(self, message: str, /, **kwargs: object):
        self.message = message
        self.kwargs = kwargs

    @override
    def __str__(self) -> str:
        return (
            f"{reset_all}{bright}{self.message}{reset_all}"
            + f"{white}{dim} >>> {reset_all}{
                colorize(
                    json.dumps(self.kwargs, ensure_ascii=False, default=json_default)
                )
            }"
            + f"{reset_all}"
        )


SL = StructLog
