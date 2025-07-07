#!/bin/env python

import sys
import json
from datetime import datetime

current_msg = []


def pop_brace(msg: str) -> tuple[str, str]:
    if msg.startswith("["):
        pos = msg.find("]")
        if pos == -1:
            return msg, ""
        return msg[pos + 1 :], msg[1:pos]
    return msg, ""


def preproc(msg: str) -> str:
    if msg.startswith(" - "):
        return msg[3:]
    return msg


def flush():
    global current_msg
    msg = "".join(current_msg).strip()
    if not msg:
        return
    data = {}
    msg, time = pop_brace(msg)
    if time:
        data["time"] = time
        data["time_unix"] = datetime.strptime(time, "%Y-%m-%d %H:%M:%S,%f").timestamp()
        msg, name = pop_brace(msg)
        if name:
            data["name"] = name
            msg, level = pop_brace(msg)
            if level:
                data["level"] = level
    pos = msg.find(" >>> ")
    if pos >= 0:
        msg, obj = msg[:pos], msg[pos + 5 :]
        obj = json.loads(obj)
        data["args"] = obj
    data["msg"] = preproc(msg)
    print(data)

    current_msg = []


for line in sys.stdin.readlines():
    if line.startswith("["):
        flush()
    current_msg.append(line)

flush()
