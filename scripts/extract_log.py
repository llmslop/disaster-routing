import json
import sys
from datetime import datetime
from typing import IO


def collect_log(io: IO[str]):
    current_msg: list[str] = []
    output: list[object] = []

    def flush():
        def pop_brace(msg: str) -> tuple[str, str]:
            if msg.startswith("["):
                pos = msg.find("]")
                if pos == -1:
                    return msg, ""
                return msg[pos + 1 :], msg[1:pos]
            return msg, ""

        def preproc(msg: str) -> str:
            return msg[3:] if msg.startswith(" - ") else msg

        msg = "".join(current_msg).strip()
        if not msg:
            return
        data: object = {}
        msg, time = pop_brace(msg)
        if time:
            data["time"] = time
            data["time_unix"] = datetime.strptime(
                time, "%Y-%m-%d %H:%M:%S,%f"
            ).timestamp()
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
        output.append(data)
        current_msg.clear()

    for line in io.readlines():
        if line.startswith("["):
            flush()
        current_msg.append(line)
    flush()

    return output


if __name__ == "__main__":
    json.dump(collect_log(sys.stdin), sys.stdout)
