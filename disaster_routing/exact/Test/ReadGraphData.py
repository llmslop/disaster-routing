import sys
import networkx as nx
from lxml import etree
import io


def yap(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)


# Đường dẫn đến tệp GraphML
data = sys.stdin.read()

# Đọc GraphML bằng NetworkX
G = nx.read_graphml(io.StringIO(data))

# Đọc GraphML bằng lxml để lấy thông tin từ XML
root = etree.fromstring(data.encode())

# Định nghĩa namespace của GraphML (cần thiết để truy vấn XML)
ns = {"y": "http://www.yworks.com/xml/graphml"}

# 📌 Mảng lưu kết quả
V = []  # Danh sách node hợp lệ
A = []  # Danh sách liên kết hợp lệ (source, target, weight)
D = []  # Danh sách node hình vuông hợp lệ
Z = []  # Danh sách nhóm các node hợp lệ

# 📌 Bước 1: Tạo ánh xạ {node_id: label} để tra cứu nhanh
node_labels = {}
for node, data in G.nodes(data=True):
    label = data.get("label", node)
    try:
        label = int(label)  # Chuyển nhãn về số nguyên nếu có thể
    except ValueError:
        continue  # Bỏ qua nếu không thể chuyển thành số nguyên
    node_labels[node] = label

# Lọc bỏ các node có nhãn dạng "n**"
filtered_nodes = {node: label for node, label in node_labels.items()}
V = sorted(filtered_nodes.values())  # Sắp xếp V tăng dần

# 📌 Bước 2: Tạo ánh xạ {edge_id: trọng số} từ XML
edge_weights = {}
for edge in root.findall(".//edge", root.nsmap):
    source = edge.get("source")
    target = edge.get("target")

    # Tìm y:EdgeLabel để lấy trọng số
    edge_label = edge.find(".//y:EdgeLabel", ns)
    weight = edge_label.text if edge_label is not None else "Không có trọng số"

    # Chuyển trọng số thành số nếu có thể
    try:
        weight = int(weight)
    except ValueError:
        continue  # Bỏ qua nếu trọng số không phải số nguyên

    edge_weights[(source, target)] = weight

# 📌 Bước 3: Lấy danh sách các node có hình vuông (chỉ lấy node hợp lệ)
square_nodes = {}
for node in root.findall(".//node", root.nsmap):
    node_id = node.get("id")
    shape_node = node.find(".//y:ShapeNode", ns)  # Tìm phần tử ShapeNode
    if shape_node is not None:
        shape = shape_node.find(".//y:Shape", ns)  # Kiểm tra hình dạng
        if shape is not None and shape.get("type") == "rectangle":
            label = node_labels.get(node_id)  # Lấy nhãn của node
            if label is not None:
                square_nodes[node_id] = label

D = sorted(square_nodes.values())  # Sắp xếp D tăng dần

# 📌 Bước 4: Tạo danh sách chứa các nhóm và các node bên trong nhóm (chỉ lấy node hợp lệ)
groups = []
for group in root.findall(".//node[@yfiles.foldertype='group']", root.nsmap):
    group_id = group.get("id")  # Lấy ID của group

    # Lấy danh sách node thuộc group này
    node_list = []
    subgraph = group.find("graph", root.nsmap)  # Tìm subgraph bên trong group

    if subgraph is not None:
        for node in subgraph.findall("node", root.nsmap):
            node_id = node.get("id")
            node_label = node_labels.get(node_id)  # Thay ID bằng nhãn
            if node_label is not None:
                node_list.append(node_label)

    # Lưu vào danh sách nhóm nếu có node hợp lệ
    if node_list:
        groups.append(sorted(node_list))  # Sắp xếp từng nhóm

Z = sorted(
    groups, key=lambda x: x[0] if x else float("inf")
)  # Sắp xếp nhóm theo node nhỏ nhất

# 📌 Bước 5: Lưu danh sách các liên kết (chỉ lấy liên kết giữa các node hợp lệ)
for source, target in G.edges():
    if source not in filtered_nodes or target not in filtered_nodes:
        continue  # Bỏ qua nếu 1 trong 2 node không hợp lệ

    # Lấy trọng số từ ánh xạ edge_weights
    weight = edge_weights.get((source, target))
    if weight is None:
        continue  # Bỏ qua nếu không có trọng số hợp lệ

    # Lấy label của source và target
    source_label = node_labels[source]
    target_label = node_labels[target]

    A.append((source_label, target_label, weight))  # Lưu vào mảng A
    # A.append((target_label, source_label, weight))

A = sorted(A)  # Sắp xếp A tăng dần theo (source, target, weight)

# 📌 Kết quả
yap("\n📍 Mảng V (Danh sách nhãn của các node hợp lệ):")
yap(V)

yap("\n🔗 Mảng A (Danh sách các liên kết hợp lệ):")
yap(A)

yap("\n🟦 Mảng D (Danh sách các node hình vuông hợp lệ):")
yap(D)

yap("\n📌 Mảng Z (Danh sách nhóm các node hợp lệ):")
yap(Z)


import pandas as pd

# 📌 Tạo DataFrame cho từng loại dữ liệu
df_nodes = pd.DataFrame(V, columns=["Node"])
df_edges = pd.DataFrame(A, columns=["Source", "Target", "Weight"])
df_datacenter = pd.DataFrame(D, columns=["Datacenter"])
df_deadzone = pd.DataFrame(
    {
        "Group": [i + 1 for i in range(len(Z))],
        "Nodes": [", ".join(map(str, group)) for group in Z],
    }
)

# 📌 Lưu vào file Excel
file_output = "COST239.xlsx"
with pd.ExcelWriter(sys.stdout.buffer, engine="xlsxwriter") as writer:
    df_nodes.to_excel(writer, sheet_name="Nodes", index=False)
    df_edges.to_excel(writer, sheet_name="Edges", index=False)
    df_datacenter.to_excel(writer, sheet_name="Datacenters", index=False)
    df_deadzone.to_excel(writer, sheet_name="Deadzones", index=False)

yap(f"\n✅ Dữ liệu đã được lưu vào {file_output}")
