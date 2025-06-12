from typing import cast


class Request:
    source: int
    max_path_count: int
    content_id: int
    bpsk_fs_count: int

    def __init__(
        self, source: int, max_path_count: int, content_id: int, bpsk_fs_count: int
    ):
        self.source = source
        self.max_path_count = max_path_count
        self.content_id = content_id
        self.bpsk_fs_count = bpsk_fs_count

    @staticmethod
    def from_json(data: dict[str, object]) -> "Request":
        return Request(
            cast(int, data["source"]),
            cast(int, data["max_path_count"]),
            cast(int, data["content_id"]),
            cast(int, data["bpsk_fs_count"]),
        )

    def to_json(self):
        return self.__dict__
