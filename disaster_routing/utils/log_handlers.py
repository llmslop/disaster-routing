from logging import FileHandler, LogRecord
import re
from typing import override


class DecoloringFileHandler(FileHandler):
    ansi_escape: re.Pattern[str]

    def __init__(self, filename, mode="a", encoding=None, delay=False):
        super().__init__(filename, mode, encoding, delay)
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    @override
    def format(self, record: LogRecord) -> str:
        return self.ansi_escape.sub("", super().format(record))
