import pandas as pd
import random
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value

import time

# Bắt đầu đo thời gian chạy
start_time = time.time()

# Khởi tạo một mô hình tối ưu hóa
mdl = LpProblem("CDP_Model", LpMinimize)

# Đọc file Excel
file_path = "NSFNET.xlsx"

# Đọc từng sheet
df_nodes = pd.read_excel(file_path, sheet_name="Nodes")
df_edges = pd.read_excel(file_path, sheet_name="Edges")
df_datacenter = pd.read_excel(file_path, sheet_name="Datacenters")
df_deadzone = pd.read_excel(file_path, sheet_name="Deadzones")

# Tập hợp V (Danh sách Nodes hợp lệ)
V = sorted(df_nodes["Node"].tolist())

# Tập hợp A (Danh sách Liên kết hợp lệ)
A = []
for _, row in df_edges.iterrows():
    edge = (row["Source"], row["Target"], row["Weight"])
    reverse_edge = (row["Target"], row["Source"], row["Weight"])

    A.append(edge)
    A.append(reverse_edge)  # Thêm cạnh đảo ngược


# Tập hợp D (Danh sách Datacenter - Nodes hình vuông)
D = sorted(df_datacenter["Datacenter"].tolist())

# Tập hợp Z (Danh sách Deadzones - Nhóm liên kết bị ảnh hưởng)
Z = []
for _, row in df_deadzone.iterrows():
    nodes_value = row["Nodes"]
    if isinstance(nodes_value, str):  # Nếu là chuỗi, chia tách thành danh sách
        affected_nodes = set(map(int, nodes_value.split(",")))
    else:  # Nếu không phải chuỗi (số nguyên), chuyển thành tập hợp chỉ chứa số đó
        affected_nodes = {int(nodes_value)}

    affected_edges = [
        edge for edge in A if edge[0] in affected_nodes or edge[1] in affected_nodes
    ]

    if affected_edges:
        Z.append(affected_edges)


# Kết quả
print("\nTập hợp V (Danh sách Nodes):")
print(V)

print("\nTập hợp A (Danh sách Liên kết):")
print(A)

print("\nTập hợp D (Danh sách Datacenter - Nodes Hình Vuông):")
print(D)

print("\nTập hợp Z (Danh sách Deadzones - Nhóm Liên kết bị ảnh hưởng):")
print(Z)

C = [0000, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]
K = 3

S = [i for i in range(1, 301)]

M = [
    ({"name": "BPSK", "hm": 9600, "Tm": 12.5}),
    ({"name": "QPSK", "hm": 4800, "Tm": 25}),
    ({"name": " 8-QAM", "hm": 2400, "Tm": 37.5}),
    ({"name": "16-QAM", "hm": 1200, "Tm": 50}),
]


# Các yêu cầu
# R = [({'sr': V[10], 'zr': Z[10], '|kr|': 3, 'cr': C[0], 'φr': 50/12.5})]
def generate_requests(num_requests, V, Z, C):
    R = []

    for _ in range(num_requests):
        sr = random.choice(
            [v for v in V if v not in D]
        )  # Chọn ngẫu nhiên một nút nguồn
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


# Ví dụ sử dụng:
num_requests = 10  # Số lượng yêu cầu cần tạo
R = generate_requests(num_requests, V, Z, C)
print("Tập hợp R :")
print(R)

# Biến nhị phân
p_kra = {
    (k, r_idx, a_idx): LpVariable(f"p_{k}_{r_idx}_{a_idx}", cat="Binary")
    for r_idx, r in enumerate(R)
    for a_idx, a in enumerate(A)
    for k in range(1, r["|kr|"] + 1)
}

lambda_krd = {
    (k, r_idx, d): LpVariable(f"lambda_{k}_{r_idx}_{d}", cat="B..inary")
    for r_idx, r in enumerate(R)
    for d in D
    for k in range(1, r["|kr|"] + 1)
}

R_crd = {
    (c, r_idx, d): LpVariable(f"R_{c}_{r_idx}_{d}", cat="Binary")
    for c in C
    for r_idx, r in enumerate(R)
    for d in D
}

w_kr = {
    (k, r_idx): LpVariable(f"w_{k}_{r_idx}", cat="Binary")
    for r_idx, r in enumerate(R)
    for k in range(1, r["|kr|"] + 1)
}

alpha_krz = {
    (k, r_idx, z_idx): LpVariable(f"alpha_{k}_{r_idx}_{z_idx}", cat="Binary")
    for r_idx, r in enumerate(R)
    for z_idx, z in enumerate(Z)
    for k in range(1, r["|kr|"] + 1)
}

Xi_ir = {
    (i, r_idx): LpVariable(f"Xi_{i}_{r_idx}", cat="Binary")
    for r_idx, r in enumerate(R)
    for i in range(1, r["|kr|"] + 1)
}

beta_kkp_r = {
    (k, kp, r_idx): LpVariable(f"beta_{k}_{kp}_{r_idx}", cat="Binary")
    for r_idx, r in enumerate(R)
    for k in range(1, r["|kr|"] + 1)
    for kp in range(1, r["|kr|"] + 1)
}  # if k > kp}

beta_kkp_rrp = {
    (k, kp, r_idx, rp_idx): LpVariable(f"beta_{k}_{kp}_{r_idx}_{rp_idx}", cat="Binary")
    for r_idx, r in enumerate(R)
    for rp_idx, rp in enumerate(R)  # if r_idx > rp_idx
    for k in range(1, r["|kr|"] + 1)
    for kp in range(1, rp["|kr|"] + 1)
}

gamma_kkp_r = {
    (k, kp, r_idx): LpVariable(f"gamma_{k}_{kp}_{r_idx}", cat="Binary")
    for r_idx, r in enumerate(R)
    for k in range(1, r["|kr|"] + 1)
    for kp in range(1, r["|kr|"] + 1)
}  # if k > kp}

gamma_kkp_rrp = {
    (k, kp, r_idx, rp_idx): LpVariable(f"gamma_{k}_{kp}_{r_idx}_{rp_idx}", cat="Binary")
    for r_idx, r in enumerate(R)
    for rp_idx, rp in enumerate(R)  # if r_idx > rp_idx
    for k in range(1, r["|kr|"] + 1)
    for kp in range(1, rp["|kr|"] + 1)
}


b_kmr = {
    (k, m_idx, r_idx): LpVariable(f"b_{k}_{m_idx}_{r_idx}", cat="Binary")
    for r_idx, r in enumerate(R)
    for k in range(1, r["|kr|"] + 1)
    for m_idx, m in enumerate(M)
}


# Biến nguyên
phi_krm = {
    (k, r_idx, m_idx): LpVariable(
        f"phi_krm_{k}_{r_idx}_{m_idx}", lowBound=0, upBound=len(S), cat="Integer"
    )
    for r_idx, r in enumerate(R)
    for m_idx, m in enumerate(M)
    for k in range(1, r["|kr|"] + 1)
}

phi_kra = {
    (k, r_idx, a_idx): LpVariable(
        f"phi_kra_{k}_{r_idx}_{a_idx}", lowBound=0, upBound=len(S), cat="Integer"
    )
    for r_idx, r in enumerate(R)
    for a_idx, a in enumerate(A)
    for k in range(1, r["|kr|"] + 1)
}

phi_kr = {
    (k, r_idx): LpVariable(
        f"phi_kr_{k}_{r_idx}", lowBound=0, upBound=len(S), cat="Integer"
    )
    for r_idx, r in enumerate(R)
    for k in range(1, r["|kr|"] + 1)
}

g_kr = {
    (k, r_idx): LpVariable(
        f"g_kr_{k}_{r_idx}", lowBound=0, upBound=len(S) - 1, cat="Integer"
    )
    for r_idx, r in enumerate(R)
    for k in range(1, r["|kr|"] + 1)
}

MOFI = LpVariable("MOFI", lowBound=0, upBound=len(S), cat="Integer")


# RB(2): Mỗi DC chỉ có thể được gán cho working/backup path của mỗi yêu cầu 1 lần
for r_idx, r in enumerate(R):
    for k in range(1, r["|kr|"] + 1):
        # Ràng buộc: Σ lambda_{k, r, d} = w_{k, r}
        mdl += (
            lpSum(lambda_krd[k, r_idx, d] for d in D) == w_kr[k, r_idx],
            f"dc_assignment_content_placement_{k}_{r_idx}",
        )


# RB(3): Ràng buộc giới hạn số lượng DC lưu trữ nội dung cho mỗi yêu cầu
for r_idx, r in enumerate(R):
    for c in C:
        # Ràng buộc: 2 ≤ Σ R_{c, r, d} ≤ |kr|, ∀r
        # mdl += lpSum(R_crd[c, r_idx, d] for d in D) >= 2, f"content_storage_lower_bound_{c}_{r_idx}"
        # mdl += lpSum(R_crd[c, r_idx, d] for d in D) <= r['|kr|'], f"content_storage_upper_bound_{c}_{r_idx}"
        mdl += (
            lpSum(R_crd[c, r_idx, d] for d in D) == K,
            f"content_storage_exact_K_{c}_{r_idx}",
        )

# RB(4): Ràng buộc đảm bảo rằng các DC được phân bổ khác nhau cho các đường dẫn
for r_idx, r in enumerate(R):
    for d in D:
        # Ràng buộc: Σ λ_{k, r, d} ≤ R_{c, r, d}, ∀r, ∀d
        mdl += (
            lpSum(lambda_krd[k, r_idx, d] for k in range(1, r["|kr|"] + 1))
            <= R_crd[c, r_idx, d],
            f"dc_unique_assignment_{r_idx}_{d}",
        )


# RB(5): Ràng buộc Flow-conservation constraints
def get_out_links(node):
    # Liên kết đi ra từ node
    outgoing_links = [link for link in A if link[0] == node]
    return outgoing_links


def get_in_links(node):
    # Liên kết đi vào node
    incoming_links = [link for link in A if link[1] == node]
    return incoming_links


def get_in_and_out_links(node):
    # Liên kết đi vào node
    incoming_links = [link for link in A if link[0] == node or link[1] == node]
    return incoming_links


# RB(5): Ràng buộc bảo toàn luồng
for r_idx, r in enumerate(R):
    for k in range(1, r["|kr|"] + 1):
        for v in V:
            # Lấy tất cả liên kết ra và vào
            outgoing_links = get_out_links(v)
            incoming_links = get_in_links(v)

            # Tính tổng dòng chảy ĐÚNG CÁCH
            outgoing_flow = lpSum(p_kra[k, r_idx, A.index(a)] for a in outgoing_links)
            incoming_flow = lpSum(p_kra[k, r_idx, A.index(a)] for a in incoming_links)

            # Ràng buộc bảo toàn luồng
            if v == r["sr"]:
                mdl += (
                    (outgoing_flow - incoming_flow) == w_kr[k, r_idx],
                    f"flow_conservation_source_{k}_{r_idx}",
                )
            elif v in D:
                mdl += (
                    (outgoing_flow - incoming_flow) == -lambda_krd[k, r_idx, v],
                    f"flow_conservation_dc_{k}_{r_idx}_{v}",
                )
            else:
                mdl += (
                    (outgoing_flow - incoming_flow) == 0,
                    f"flow_conservation_transit_{k}_{r_idx}_{v}",
                )


# RB(6, 7): Ràng buộc Disaster-zone-disjoint path constraints (6) và (7)
for r_idx, r in enumerate(R):
    for k in range(1, r["|kr|"] + 1):  # Duyệt qua tất cả các đường dẫn của yêu cầu r
        for z_idx, z in enumerate(Z):  # Duyệt qua tất cả các khu vực thảm họa
            # Ràng buộc (6): α_{k, r, z} ≤ Σ p_{k, r, a}, a ∈ z
            mdl += (
                alpha_krz[k, r_idx, z_idx]
                <= lpSum(p_kra[k, r_idx, a_idx] for a_idx, a in enumerate(z)),
                f"disaster_zone_constraint6_{k}_{r_idx}_{z_idx}",
            )

            # Ràng buộc (7): α_{k, r, z} ≥ p_{k, r, a}, ∀a ∈ z
            for a_idx, a in enumerate(z):
                mdl += (
                    alpha_krz[k, r_idx, z_idx] >= p_kra[k, r_idx, a_idx],
                    f"disaster_zone_constraint7_{k}_{r_idx}_{z_idx}_{a_idx}",
                )


# RB(8): Đảm bảo rằng các đường truyền không đi qua cùng một vùng DZ (trừ vùng chứa nút nguồn)
for r_idx, r in enumerate(R):
    zr = r["zr"]  # Khu vực thảm họa đặc biệt của yêu cầu r
    for k in range(1, r["|kr|"] + 1):  # Duyệt qua các đường dẫn của yêu cầu r
        for z_idx, z in enumerate(Z):
            if z != zr:  # Kiểm tra xem khu vực thảm họa z có phải là z_r không
                # Ràng buộc α_{k, r, z} ≤ 1 cho mọi k, r và z ∉ zr
                mdl += (
                    lpSum((alpha_krz[k, r_idx, z_idx] for k in range(1, r["|kr|"] + 1)))
                    <= 1,
                    f"disaster_zone_disjoint_constraint_{k}_{r_idx}_{z_idx}",
                )

# Ràng buộc (9): Modulation format selection constraint
h_max = 9600
for r_idx, r in enumerate(R):
    for m_idx, m in enumerate(M):
        for k in range(1, r["|kr|"] + 1):
            mdl += (
                lpSum(a[2] * p_kra[k, r_idx, a_idx] for a_idx, a in enumerate(A))
                <= m["hm"] + h_max * (1 - b_kmr[k, m_idx, r_idx]),
                f"modulation_format_selection_{k}_{r_idx}_{m_idx}",
            )


# Ràng buộc (10): Ensure only one modulation format is selected for each path
for r_idx, r in enumerate(R):
    for k in range(1, r["|kr|"] + 1):
        mdl += (
            lpSum(b_kmr[k, m_idx, r_idx] for m_idx, m in enumerate(M))
            <= w_kr[k, r_idx],
            f"one_modulation_format_per_path_{k}_{r_idx}",
        )


# Ràng buộc (11): FS assigned for each request
for r_idx, r in enumerate(R):
    for k in range(1, r["|kr|"] + 1):
        mdl += (
            phi_kr[k, r_idx]
            == lpSum(phi_krm[k, r_idx, m_idx] for m_idx, m in enumerate(M)),
            f"fs_assigned_{k}_{r_idx}",
        )


# Ràng buộc (12): No FS assigned for non-selected modulation formats
for r_idx, r in enumerate(R):
    for k in range(1, r["|kr|"] + 1):
        for m_idx, m in enumerate(M):
            mdl += (
                phi_krm[k, r_idx, m_idx] <= b_kmr[k, m_idx, r_idx] * len(S),
                f"fs_assignment_constraint_{k}_{r_idx}_{m_idx}",
            )


# Ràng buộc (13): Ngăn tạo đường dẫn nếu k-th path không được chọn
for r_idx, r in enumerate(R):
    for a_idx, a in enumerate(A):
        for k in range(1, r["|kr|"] + 1):
            mdl += (
                p_kra[k, r_idx, a_idx] <= w_kr[k, r_idx],
                f"path_selection_constraint_{k}_{r_idx}_{a_idx}",
            )


# Ràng buộc (14): Số lượng đường dẫn hoạt động cho mỗi yêu cầu
for r_idx, r in enumerate(R):
    mdl += (
        lpSum(w_kr[i, r_idx] for i in range(1, r["|kr|"]))
        == lpSum(i * Xi_ir[i, r_idx] for i in range(1, r["|kr|"])),
        f"number_of_working_paths_{r_idx}",
    )


# Ràng buộc (15): Đảm bảo chỉ một số đường dẫn hoạt động được chọn
for r_idx, r in enumerate(R):
    mdl += (
        lpSum(Xi_ir[i, r_idx] for i in range(1, r["|kr|"])) == 1,
        f"single_working_path_selection_{r_idx}",
    )


# Ràng buộc (16): Đường dẫn đầu tiên là hoạt động, đường dẫn cuối là dự phòng
for r_idx, r in enumerate(R):
    mdl += w_kr[1, r_idx] == 1, f"first_path_active_{r_idx}"

    mdl += w_kr[r["|kr|"], r_idx] == 1, f"last_path_backup_{r_idx}"


# Ràng buộc (17): Ưu tiên các đường dẫn có chỉ số nhỏ hơn
for r_idx, r in enumerate(R):
    for k in range(1, r["|kr|"] - 1):
        mdl += w_kr[k, r_idx] >= w_kr[k + 1, r_idx], f"path_preference_{k}_{r_idx}"


# Ràng buộc (18): Tổng FSs phải đủ để phục vụ yêu cầu
for r_idx, r in enumerate(R):
    for k in range(1, r["|kr|"] + 1):
        mdl += (
            lpSum(phi_krm[k, r_idx, m_idx] * m["Tm"] for m_idx, m in enumerate(M))
            + (1 - w_kr[k, r_idx]) * r["φr"]
            >= r["φr"]
            * lpSum((Xi_ir[i, r_idx] * (1 / i)) for i in range(1, r["|kr|"])),
            f"fs_assignment_sufficient_{k}_{r_idx}",
        )


# Ràng buộc (19): Số FS trên liên kết không vượt quá |S| nếu liên kết được chọn
for r_idx, r in enumerate(R):
    for k in range(1, r["|kr|"] + 1):
        for a_idx, a in enumerate(A):
            mdl += (
                phi_kra[k, r_idx, a_idx] <= p_kra[k, r_idx, a_idx] * len(S),
                f"fs_limit_on_link_{k}_{r_idx}_{a_idx}",
            )


# Ràng buộc (20): Số FS trên liên kết không vượt quá tổng số FS của đường dẫn
for r_idx, r in enumerate(R):
    for k in range(1, r["|kr|"] + 1):
        for a_idx, a in enumerate(A):
            mdl += (
                phi_kra[k, r_idx, a_idx] <= phi_kr[k, r_idx],
                f"fs_link_less_than_path_{k}_{r_idx}_{a_idx}",
            )


# Ràng buộc (21): FS trên liên kết bằng FS của đường dẫn nếu liên kết được chọn
for r_idx, r in enumerate(R):
    for k in range(1, r["|kr|"] + 1):
        for a_idx, a in enumerate(A):
            mdl += (
                phi_kra[k, r_idx, a_idx]
                >= phi_kr[k, r_idx] - len(S) * (1 - p_kra[k, r_idx, a_idx]),
                f"fs_link_equals_path_if_selected_{k}_{r_idx}_{a_idx}",
            )


# RB(22) :
for r_idx, r in enumerate(R):
    for a_idx, a in enumerate(A):
        for k in range(1, r["|kr|"] + 1):
            for kp in range(1, r["|kr|"] + 1):
                if k > kp:
                    mdl += (
                        p_kra[k, r_idx, a_idx] + p_kra[kp, r_idx, a_idx] - 1
                        <= gamma_kkp_r[k, kp, r_idx],
                        f"shared_link_same_request_{k}_{kp}_{r_idx}_{a_idx}",
                    )


# RB(23):
for r_idx, r in enumerate(R):
    for k in range(1, r["|kr|"] + 1):
        for kp in range(1, r["|kr|"] + 1):
            if k > kp:
                mdl += (
                    gamma_kkp_r[k, kp, r_idx] == gamma_kkp_r[kp, k, r_idx],
                    f"symmetric_gamma_same_request_{k}_{kp}_{r_idx}",
                )


# Sửa phần RB(24): Đảm bảo gamma_kkp_rrp được khởi tạo cho tất cả các tổ hợp hợp lệ
for r_idx, r in enumerate(R):
    for rp_idx, rp in enumerate(R):
        if r_idx != rp_idx:  # Đảm bảo r_idx và rp_idx khác nhau
            for a_idx, a in enumerate(A):
                for k in range(1, r["|kr|"] + 1):
                    for kp in range(1, rp["|kr|"] + 1):
                        # Đảm bảo khóa tồn tại trong p_kra và gamma_kkp_rrp
                        if (k, r_idx, a_idx) in p_kra and (kp, rp_idx, a_idx) in p_kra:
                            if (k, kp, r_idx, rp_idx) not in gamma_kkp_rrp:
                                gamma_kkp_rrp[k, kp, r_idx, rp_idx] = LpVariable(
                                    f"gamma_{k}_{kp}_{r_idx}_{rp_idx}", cat="Binary"
                                )
                            # Thêm ràng buộc
                            mdl += (
                                p_kra[k, r_idx, a_idx] + p_kra[kp, rp_idx, a_idx] - 1
                                <= gamma_kkp_rrp[k, kp, r_idx, rp_idx],
                                f"shared_link_diff_request_{k}_{kp}_{r_idx}_{rp_idx}_{a_idx}",
                            )


# RB(25): Đảm bảo tính đối xứng trong gamma_kkp_rrp
for r_idx, r in enumerate(R):
    for rp_idx, rp in enumerate(R):
        if r_idx != rp_idx:
            for k in range(1, r["|kr|"] + 1):
                for kp in range(1, rp["|kr|"] + 1):
                    # Đảm bảo khóa tồn tại trong gamma_kkp_rrp
                    if (k, kp, r_idx, rp_idx) not in gamma_kkp_rrp:
                        gamma_kkp_rrp[k, kp, r_idx, rp_idx] = LpVariable(
                            f"gamma_{k}_{kp}_{r_idx}_{rp_idx}", cat="Binary"
                        )
                    if (kp, k, rp_idx, r_idx) not in gamma_kkp_rrp:
                        gamma_kkp_rrp[kp, k, rp_idx, r_idx] = LpVariable(
                            f"gamma_{kp}_{k}_{rp_idx}_{r_idx}", cat="Binary"
                        )
                    # Thêm ràng buộc
                    mdl += (
                        gamma_kkp_rrp[k, kp, r_idx, rp_idx]
                        == gamma_kkp_rrp[kp, k, rp_idx, r_idx],
                        f"symmetric_gamma_diff_request_{k}_{kp}_{r_idx}_{rp_idx}",
                    )


# RB(26):
for r_idx, r in enumerate(R):
    for k in range(1, r["|kr|"] + 1):
        for kp in range(1, r["|kr|"] + 1):
            if k > kp:
                mdl += (
                    beta_kkp_r[k, kp, r_idx] + beta_kkp_r[kp, k, r_idx] == 1,
                    f"compare_fs_same_request_{k}_{kp}_{r_idx}",
                )


# RB(27):
for r_idx, r in enumerate(R):
    for rp_idx, rp in enumerate(R):
        if r_idx > rp_idx:
            for k in range(1, r["|kr|"] + 1):
                for kp in range(1, rp["|kr|"] + 1):
                    mdl += (
                        beta_kkp_rrp[k, kp, r_idx, rp_idx]
                        + beta_kkp_rrp[kp, k, rp_idx, r_idx]
                        == 1,
                        f"compare_fs_diff_request_{k}_{kp}_{r_idx}_{rp_idx}",
                    )

# Ràng buộc (28)
for r_idx, r in enumerate(R):
    for k in range(1, r["|kr|"] + 1):
        mdl += g_kr[k, r_idx] + phi_kr[k, r_idx] <= MOFI, f"max_fs_index_{k}_{r_idx}"


# RB(29):
# Định nghĩa một hằng số lớn M
M = 1e6
for r_idx, r in enumerate(R):
    for k in range(1, r["|kr|"] + 1):
        for kp in range(1, r["|kr|"] + 1):
            if k != kp:
                mdl += (
                    g_kr[k, r_idx] + phi_kr[k, r_idx] - g_kr[kp, r_idx]
                    <= M * (2 - gamma_kkp_r[k, kp, r_idx] - beta_kkp_r[k, kp, r_idx]),
                    f"spectrum_conflict_same_request_{k}_{kp}_{r_idx}",
                )

# RB(30):
for r_idx, r in enumerate(R):
    for rp_idx, rp in enumerate(R):
        if r_idx != rp_idx:
            for k in range(1, r["|kr|"] + 1):
                for kp in range(1, rp["|kr|"] + 1):
                    mdl += (
                        g_kr[k, r_idx] + phi_kr[k, r_idx] - g_kr[kp, rp_idx]
                        <= M
                        * (
                            2
                            - gamma_kkp_rrp[k, kp, r_idx, rp_idx]
                            - beta_kkp_rrp[k, kp, r_idx, rp_idx]
                        ),
                        f"spectrum_conflict_different_requests_{k}_{kp}_{r_idx}_{rp_idx}",
                    )


# --- Hàm mục tiêu ---
# Objective function
theta_1 = 1
theta_2 = 1
objective = (
    theta_1
    * lpSum(
        phi_kra[k, r_idx, a_idx]
        for r_idx, r in enumerate(R)
        for a_idx, a in enumerate(A)
        for k in range(1, r["|kr|"] + 1)
    )
    + theta_2 * MOFI
)
mdl += objective

# Giải bài toán
# mdl.print_information()
mdl.solve()


# Phần code của bạn để giải quyết mô hình và in kết quả
if mdl.status == 1:
    result = "------------------------- Solution Found: -------------------------\n"
    result += f"Objective value: {value(mdl.objective)}\n"
    for v in mdl.variables():
        if v.varValue > 0:
            result += f"{v.name} = {v.varValue}\n"

    # In giá trị của biến MOFI
    if "MOFI" in mdl.variablesDict():
        MOFI_value = value(mdl.variablesDict()["MOFI"])
        result += f"MOFI = {MOFI_value}\n"
    else:
        result += "MOFI variable not found in the model.\n"

else:
    result = "No solution found.\n"

# In các liên kết thuộc về mỗi đường dẫn
for r_idx, r in enumerate(R):
    result += f"\nYêu cầu {r_idx + 1} (Source: {r['sr']}):\n"
    for k in range(1, r["|kr|"] + 1):
        selected_links = [
            A[a_idx] for a_idx, a in enumerate(A) if value(p_kra[k, r_idx, a_idx]) == 1
        ]
        result += f"   🔹 Đường dẫn {k}: {selected_links}\n"

# Kết thúc đo thời gian chạy
end_time = time.time()
execution_time = end_time - start_time

# Thêm thời gian chạy vào kết quả
result += f"\nThời gian chạy chương trình: {execution_time:.2f} giây\n\n\n"

# Lưu kết quả vào file txt
with open("results.txt", "a", encoding="utf-8") as file:
    file.write(result)


# Print results and generate graphs
if mdl.status == 1:
    print("Solution Found:")
    print(f"Objective value: {value(mdl.objective)}")

    # Create the directed graph
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.DiGraph()
    G.add_nodes_from(V)
    for a in A:
        G.add_edge(a[0], a[1], weight=a[2])

    # For each request, plot the graph with highlighted paths
    for r_idx, r in enumerate(R):
        used_edges = []
        for k in range(1, r["|kr|"] + 1):
            for a_idx, a in enumerate(A):
                var = p_kra.get((k, r_idx, a_idx), None)
                if var and var.varValue == 1:
                    used_edges.append((a[0], a[1]))

        # Draw the graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)  # Consistent layout

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500)
        nx.draw_networkx_nodes(G, pos, nodelist=D, node_color="green", node_size=700)
        nx.draw_networkx_labels(G, pos)

        # Draw edges
        all_edges = list(G.edges())
        edge_colors = ["red" if edge in used_edges else "gray" for edge in all_edges]
        nx.draw_networkx_edges(
            G, pos, edgelist=all_edges, edge_color=edge_colors, arrows=True
        )

        plt.title(f"Paths for Request {r_idx} (Source: {r['sr']})")
        plt.savefig(f"request_{r_idx}_paths.png")
        plt.close()
        print(f"Saved graph for request {r_idx} as request_{r_idx}_paths.png")
else:
    print("No solution found.")

