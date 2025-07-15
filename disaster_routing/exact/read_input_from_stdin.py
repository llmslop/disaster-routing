import json
from typing import IO
import pandas as pd
import sys

from disaster_routing.instances.instance import Instance

import random

# remove stochasticity
random.seed(42)


def read_input_from_stdin_xlsx(
    inp: IO[bytes] = sys.stdin.buffer,
) -> tuple[
    list[int], list[tuple[int, int, int]], list[int], list[list[tuple[int, int, int]]]
]:
    excel = pd.ExcelFile(inp)
    df_nodes = pd.read_excel(excel, sheet_name="Nodes")
    df_edges = pd.read_excel(excel, sheet_name="Edges")
    df_datacenter = pd.read_excel(excel, sheet_name="Datacenters")
    df_deadzone = pd.read_excel(excel, sheet_name="Deadzones")

    # Tập hợp V (Danh sách Nodes hợp lệ)
    V = sorted(df_nodes["Node"].tolist())

    # Tập hợp A (Danh sách Liên kết hợp lệ)
    A = []
    for _, row in df_edges.iterrows():
        edge = (int(row["Source"]), int(row["Target"]), int(row["Weight"]))
        reverse_edge = (int(row["Target"]), int(row["Source"]), int(row["Weight"]))

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
            affected_edges.sort()
            Z.append(affected_edges)

    A.sort()
    Z.sort()
    return V, A, D, Z


def read_input_from_stdin_inst(
    inp: IO[bytes] = sys.stdin.buffer,
) -> tuple[
    list[int], list[tuple[int, int, int]], list[int], list[list[tuple[int, int, int]]]
]:
    inst = Instance.from_json(json.load(inp))
    top = inst.topology
    graph = top.graph

    V = sorted(v for v in graph.nodes)
    A: list[tuple[int, int, int]] = []
    for u, v in graph.edges:
        A.append((u, v, graph.edges[u, v]["weight"]))
    D = [2, 5, 9]  # TODO: make this part of the instance maybe?
    Z: list[list[tuple[int, int, int]]] = []
    for dz in top.dzs:
        affected_edges = [e for e in A if e[0] in dz.nodes or e[1] in dz.nodes]
        if affected_edges:
            affected_edges.sort()
            Z.append(affected_edges)
    A.sort()
    Z.sort()
    return V, A, D, Z


if __name__ == "__main__":
    with open("res/Graph/NSFNET.xlsx", "rb") as f:
        nsfnet_1 = read_input_from_stdin_xlsx(f)
    with open("instances/nsfnet-0010-01.json", "rb") as f:
        nsfnet_2 = read_input_from_stdin_inst(f)

    n1 = json.dumps(nsfnet_1, sort_keys=True)
    n2 = json.dumps(nsfnet_2, sort_keys=True)

    print("NSFNET from XLSX:")
    print(n1)
    print("NSFNET from JSON instance:")
    print(n2)

    if n1 != n2:
        print("Different representations!")
    else:
        print("NSFNET repsresentations are the same!")

read_input_from_stdin = read_input_from_stdin_inst
