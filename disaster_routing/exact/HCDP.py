import random
import networkx as nx
import matplotlib.pyplot as plt

from .read_input_from_stdin import read_input_from_stdin

V, A, D, Z = read_input_from_stdin()

# Các tham số khác
C = [0000, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]  # Nội dung
K = 2  # Số lượng DC được gán cho mỗi nội dung
S = [i for i in range(1, 301)]  # Tập hợp các FS (Frequency Slots)

# Định dạng điều chế
M = [
    {"name": "16-QAM", "hm": 1200, "Tm": 50},
    {"name": "8-QAM", "hm": 2400, "Tm": 37.5},
    {"name": "QPSK", "hm": 4800, "Tm": 25},
    {"name": "BPSK", "hm": 9600, "Tm": 12.5},
]


# Hàm tạo yêu cầu
def generate_requests(num_requests, V, Z, C):
    R = []

    for _ in range(num_requests):
        sr = random.choice(V)  # Chọn ngẫu nhiên một nút nguồn
        zr = max(
            (z for z in Z if any(sr in edge[:2] for edge in z)),
            key=lambda z: sum(1 for edge in z if sr in edge[:2]),
            default=[],
        )  # Chọn vùng thảm họa chứa sr
        kr = int(
            sum(1 for edge in A if edge[0] == sr or edge[1] == sr) / 2
        )  # Bậc của nút sr
        cr = random.choice(C)  # Chọn nội dung ngẫu nhiên từ tập C
        φr = random.uniform(0.1, 125) / 12.5  # Giá trị ngẫu nhiên trong (0,125]/12.5

        R.append({"sr": sr, "zr": zr, "|kr|": kr, "cr": cr, "φr": φr})

    return R


# Tạo yêu cầu
num_requests = 10  # Số lượng yêu cầu cần tạo
R = generate_requests(num_requests, V, Z, C)
# Kết quả
print("\nTập hợp V (Danh sách Nodes):")
print(V)

print("\nTập hợp A (Danh sách Liên kết):")
print(A)

print("\nTập hợp D (Danh sách Datacenter - Nodes Hình Vuông):")
print(D)

print("\nTập hợp Z (Danh sách Deadzones - Nhóm Liên kết bị ảnh hưởng):")
print(Z)

print("\nTập hợp R :")
print(R)


# Hàm tính chi phí của một đường đi
def calculate_path_cost(path, modulation, φr, link_weights):
    """
    Tính chi phí của đường đi dựa trên trọng số của từng liên kết.

    Tham số:
    - path: Danh sách các liên kết trên đường đi (ví dụ: [(u1, v1), (u2, v2), ...]).
    - modulation: Định dạng điều chế được chọn (ví dụ: {'name': 'BPSK', 'hm': 9600, 'Tm': 12.5}).
    - φr: Yêu cầu FS của yêu cầu r.
    - link_weights: Từ điển chứa trọng số của từng liên kết (ví dụ: {(u1, v1): 1.5, (u2, v2): 2.0, ...}).

    Trả về:
    - Chi phí của đường đi (costr_p).
    """
    # Tính số lượng FS cần thiết (Φ^k_r)
    required_fs = φr / modulation["Tm"]

    # Tính tổng chi phí của đường đi: costr_p = Σ (p^k_ra * Φ^k_r * weight_a)
    costr_p = 0
    for link in path:
        weight = link_weights.get(
            link, 1.0
        )  # Mặc định trọng số là 1.0 nếu không có trong từ điển
        costr_p += required_fs * weight

    return costr_p


# Hàm tìm đường đi ngắn nhất từ DC đến nút nguồn
def find_shortest_path(G, source, target):
    try:
        path = nx.shortest_path(G, source=source, target=target, weight="weight")
        return path
    except nx.NetworkXNoPath:
        return None


# Hàm chọn định dạng điều chế dựa trên độ dài đường đi
def select_modulation(path_length, M):
    if path_length <= M[0]["hm"]:
        return M[0]

    if path_length > M[-1]["hm"]:
        return M[-1]

    for i in range(len(M) - 1):
        if M[i]["hm"] < path_length <= M[i + 1]["hm"]:
            return M[i + 1]

    return None  # Trả về None nếu không tìm thấy định dạng phù hợp


# Thuật toán HCDP
def HCDP(G, R, D, Z, M, S):
    # Khởi tạo các biến

    paths = {}
    costs = {}
    MOFI = 0
    cost_total = 0
    L_max = set()

    # Duyệt qua từng yêu cầu
    for r_idx, r in enumerate(R):
        sr = r["sr"]
        zr = r["zr"]
        kr = r["|kr|"]
        cr = r["cr"]
        φr = r["φr"]

        u_krd = set()

        # Khởi tạo đồ thị tạm thời
        G_temp = G.copy()

        # Duyệt qua từng đường đi
        for k in range(1, kr + 1):
            # Loại bỏ các liên kết và nút bị ảnh hưởng bởi DZ (trừ zr)
            if k > 1:
                for z in Z:
                    if z != zr:
                        for edge in z:
                            if G_temp.has_edge(edge[0], edge[1]):
                                G_temp.remove_edge(edge[0], edge[1])
            # Tìm đường đi ngắn nhất từ DC đến nút nguồn
            shortest_path = None
            for d in D:
                if d in u_krd:
                    break
                u_krd.add(d)

                path = find_shortest_path(G_temp, d, sr)
                if path:
                    if shortest_path is None or len(path) < len(shortest_path):
                        shortest_path = path

            if shortest_path:
                # Tính độ dài đường đi
                path_length = sum(
                    G_temp[u][v]["weight"]
                    for u, v in zip(shortest_path[:-1], shortest_path[1:])
                )

                # Chọn định dạng điều chế
                modulation = select_modulation(path_length, M)
                if modulation:
                    # Tính chi phí của đường đi
                    cost = calculate_path_cost(
                        shortest_path, modulation, φr, link_weights
                    )

                    # Lưu đường đi và chi phí
                    paths[(k, r_idx)] = shortest_path
                    costs[(k, r_idx)] = cost

                    # Cập nhật MOFI
                    MOFI = max(MOFI, cost)

                    # Cập nhật cost_total
                    cost_total += cost

                    # Cập nhật L_max (chỉ khi shortest_path có ít nhất 2 phần tử)
                    if len(shortest_path) >= 2:
                        L_max.add((shortest_path[-2], shortest_path[-1]))

    # Tối ưu hóa lại các đường đi dựa trên S lần lặp
    for s in range(S):
        for r_idx, r in enumerate(R):
            for d in D:
                for k in range(1, kr + 1):
                    if (k, r_idx) in paths:
                        path = paths[(k, r_idx)]
                        for a in path:
                            if a in L_max:
                                # Loại bỏ liên kết a và tìm đường đi mới
                                G_temp = G.copy()
                                G_temp.remove_edge(a[0], a[1])
                                new_path = find_shortest_path(G_temp, d, sr)
                                if new_path:
                                    # Tính lại chi phí
                                    new_path_length = sum(
                                        G_temp[u][v]["weight"]
                                        for u, v in zip(new_path[:-1], new_path[1:])
                                    )
                                    new_modulation = select_modulation(
                                        new_path_length, M
                                    )
                                    if new_modulation:
                                        new_cost = calculate_path_cost(
                                            new_path, new_modulation, φr
                                        )
                                        if new_cost < costs[(k, r_idx)]:
                                            # Cập nhật đường đi và chi phí
                                            paths[(k, r_idx)] = new_path
                                            costs[(k, r_idx)] = new_cost
                                            cost_total -= costs[(k, r_idx)]
                                            cost_total += new_cost
                                            L_max.remove(a)
                                            if len(new_path) >= 2:
                                                L_max.add((new_path[-2], new_path[-1]))

    return paths, costs, MOFI, cost_total, L_max


# Tạo đồ thị từ tập hợp A
G = nx.DiGraph()
for a in A:
    G.add_edge(a[0], a[1], weight=a[2])

# Chạy thuật toán HCDP
S = 100
paths, costs, MOFI, cost_total, L_max = HCDP(G, R, D, Z, M, S)

# In kết quả
print("\nKết quả thuật toán HCDP:")
for (k, r_idx), path in paths.items():
    # Lấy source node từ yêu cầu tương ứng
    source_node = R[r_idx]["sr"]
    # print(f"Yêu cầu {r_idx + 1}, Nút nguồn: {source_node}, Đường đi {k}: {path}, Chi phí: {costs[(k, r_idx)]}")
    print(f"Yêu cầu {r_idx + 1}, Nút nguồn: {source_node}, Đường đi {k}: {path}")

print(f"MOFI: {MOFI}")

# Vẽ đồ thị với các đường đi được chọn
for r_idx, r in enumerate(R):
    used_edges = []
    for k in range(1, r["|kr|"] + 1):
        if (k, r_idx) in paths:
            path = paths[(k, r_idx)]
            for u, v in zip(path[:-1], path[1:]):
                used_edges.append((u, v))

    # Vẽ đồ thị
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)  # Layout nhất quán

    # Vẽ các nút
    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500)
    nx.draw_networkx_nodes(G, pos, nodelist=D, node_color="green", node_size=700)
    nx.draw_networkx_labels(G, pos)

    # Vẽ các cạnh
    all_edges = list(G.edges())
    edge_colors = ["red" if edge in used_edges else "gray" for edge in all_edges]
    nx.draw_networkx_edges(
        G, pos, edgelist=all_edges, edge_color=edge_colors, arrows=True
    )

    plt.title(f"Paths for Request {r_idx} (Source: {r['sr']})")
    plt.savefig(f"request_{r_idx}_paths_HCDP.png")
    plt.close()
    print(f"Saved graph for request {r_idx} as request_{r_idx}_paths_HCDP.png")
