import pandas as pd
import sys


def read_input_from_stdin():
    excel = pd.ExcelFile(sys.stdin.buffer)
    df_nodes = pd.read_excel(excel, sheet_name="Nodes")
    df_edges = pd.read_excel(excel, sheet_name="Edges")
    df_datacenter = pd.read_excel(excel, sheet_name="Datacenters")
    df_deadzone = pd.read_excel(excel, sheet_name="Deadzones")

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

    return V, A, D, Z
