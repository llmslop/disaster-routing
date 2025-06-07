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
