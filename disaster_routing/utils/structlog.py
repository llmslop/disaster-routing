import json
from typing import override


class StructLog:
    message: str
    kwargs: dict[str, object]

    def __init__(self, message: str, /, **kwargs: object):
        self.message = message
        self.kwargs = kwargs

    @override
    def __str__(self) -> str:
        return "%s >>> %s" % (self.message, json.dumps(self.kwargs))


SL = StructLog
