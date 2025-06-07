import networkx as nx
import matplotlib.pyplot as plt


def parse_custom_graph(filename):
    G = nx.DiGraph()
    with open(filename, "r") as f:
        section = None
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line == "[nodes]":
                section = "nodes"
                continue
            elif line == "[links]":
                section = "links"
                continue

            if section == "nodes" and line.startswith("node ="):
                # node = ID TYPE ARCHITECTURE NUMREG
                _, rest = line.split("=", 1)
                parts = rest.strip().split()
                node_id = int(parts[0])
                node_type = parts[1]
                architecture = parts[2]
                numreg = int(parts[3])
                G.add_node(
                    node_id, type=node_type, architecture=architecture, numreg=numreg
                )

            elif section == "links" and line.startswith("-> ="):
                # -> = ORIGIN DESTINATION LENGTH
                _, rest = line.split("=", 1)
                parts = rest.strip().split()
                origin = int(parts[0])
                dest = int(parts[1])
                length = int(parts[2])
                G.add_edge(origin, dest, length=length)

    return G


G = parse_custom_graph("./res/NSFNet")
