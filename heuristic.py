import pandas as pd
from collections import deque, defaultdict

edges = pd.read_csv('data/elliptic_txs_edgelist.csv')
transactions = edges[['txId1', 'txId2']].values.tolist()

adj_list = {}                      #make adj_lst
in_neighbors = {}                  #for calculate indegree
nodes = set()                      #to store unique accounts

for u, v in transactions:
    nodes.add(u)
    nodes.add(v)
    adj_list.setdefault(u, []).append(v)
    adj_list.setdefault(v, [])
    in_neighbors.setdefault(v, []).append(u)
    in_neighbors.setdefault(u, [])

# print the graph
# for node, neighbors in adj_list.items():
#     print(f"{node} -> {neighbors}")


#degree
degree = {}
for node in nodes:
    degree[node] = (len(adj_list[node]), len(in_neighbors[node]))  #(outdegree,indegree)
    # degree[node][0] = out-degree   degree[node][1] = in-degree

#sccs = kosaraju_scc
def kosaraju_scc(nodes, adj):
    visited = set()
    finish = []

    def dfs1(start):
        stack = [(start, False)]
        while stack:
            node, done = stack.pop()
            if done:
                finish.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for nbr in adj[node]:
                if nbr not in visited:
                    stack.append((nbr, False))

    for n in nodes:
        if n not in visited:
            dfs1(n)

    rev_adj = defaultdict(list)
    for u in adj:
        for v in adj[u]:
            rev_adj[v].append(u)

    visited.clear()
    sccs = []

    def dfs2(start):
        comp = []
        stack = [start]
        visited.add(start)
        while stack:
            node = stack.pop()
            comp.append(node)
            for nbr in rev_adj[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    stack.append(nbr)
        return comp

    for node in reversed(finish):
        if node not in visited:
            sccs.append(dfs2(node))

    return sccs


sccs = kosaraju_scc(nodes, adj_list)
has_cycle = any(len(c) > 1 for c in sccs)

scc_of = {}
for comp in sccs:
    for node in comp:
        scc_of[node] = comp

max_scc = max(len(c) for c in sccs)
scc_size_score = {}
scc_density = {}
for node in nodes:
    comp = scc_of[node]
    size = len(comp)
    sub  = set(comp)
    edge_count = 0
    for u in comp:
        for v in adj_list[u]:
            if v in sub:
                edge_count += 1

    den = edge_count / (size * (size - 1)) if size > 1 else 0
    scc_size_score[node] = (size - 1) / max_scc if size > 1 else 0
    scc_density[node]    = den

# topological sort + depth
def compute_depth_kahn(nodes, adj, in_nbrs, scc_of):
    # Nodes in a non-trivial SCC (cycle) get depth = -1
    cycle_nodes = {n for n in nodes if len(scc_of[n]) > 1}

    in_deg = {n: len(in_nbrs[n]) for n in nodes}
    queue  = deque([n for n in nodes if in_deg[n] == 0 and n not in cycle_nodes])
    depth  = {n: -1 for n in nodes}

    for n in nodes:
        if n not in cycle_nodes:
            depth[n] = 0

    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v in cycle_nodes:
                continue
            depth[v] = max(depth[v], depth[u] + 1)
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)

    return depth

depth = compute_depth_kahn(nodes, adj_list, in_neighbors, scc_of)

# pagerank
d = 0.85
N = len(nodes)
pr = {node: 1 / N for node in nodes}

while True:
    pr_new = {}
    max_diff = 0
    dangling_sum = sum(pr[n] for n in nodes if degree[n][0] == 0)
    for node in nodes:
        rank_sum = 0
        for nbr in in_neighbors[node]:
            if degree[nbr][0] > 0:
                rank_sum += pr[nbr] / degree[nbr][0]

        pr_new[node] = (1 - d) / N + d * (rank_sum + dangling_sum / N)
        max_diff = max(max_diff, abs(pr_new[node] - pr[node]))

    pr = pr_new
    if max_diff < 1e-4:
        break

# clustering coefficient
adj_set = {n: set(adj_list[n]) for n in nodes}

for n in nodes:
    for nbr in in_neighbors[n]:
        adj_set[n].add(nbr)

clustering = {}

for node in nodes:
    nb = list(adj_set[node])
    k = len(nb)

    if k < 2:
        clustering[node] = 0
        continue

    links = 0
    for i in range(k):
        for j in range(i + 1, k):
            if nb[j] in adj_set[nb[i]]:
                links += 1

    clustering[node] = links / (k * (k - 1) / 2)

# motif detection
chain_score = {}
in_star_score = {}
out_star_score = {}

for node in nodes:
    in_deg = degree[node][1]
    out_deg = degree[node][0]
    chain_score[node] = in_deg * out_deg
    in_star_score[node] = in_deg * (in_deg - 1) // 2 if in_deg >= 2 else 0
    out_star_score[node] = out_deg * (out_deg - 1) // 2 if out_deg >= 2 else 0

# Triangles require cycles, so count them when has_cycle is True
# (which is also the only case they can exist)
triangle_score = {node: 0 for node in nodes}

if has_cycle:
    for u in nodes:
        for v in adj_list[u]:
            for w in adj_list[v]:
                if u in adj_list[w]:
                    triangle_score[u] += 1
                    triangle_score[v] += 1
                    triangle_score[w] += 1
# else: all zeros already — correct, since no cycles = no triangles

# bfs layering
def bfs_layering(node, adj, depth_limit=5):
    queue = deque([(node, 0)])
    visit_freq = {}

    while queue:
        curr, depth = queue.popleft()

        if depth >= depth_limit:
            continue

        for nbr in adj[curr]:
            if nbr == node:
                continue
            prev = visit_freq.get(nbr, 0)
            visit_freq[nbr] = prev + 1
            if prev == 0:               # only enqueue on first discovery
                queue.append((nbr, depth + 1))

    repeated = sum(1 for v in visit_freq.values() if v > 1)
    return min(repeated / 20.0, 1.0)


# dfs flow score
def flow_score_iter(node, adj, depth_limit=5, max_neighbors=10):
    # Use visited set instead of per-path copy — O(V+E) instead of exponential
    stack   = [(node, 0)]
    visited = {node}
    score   = 0

    while stack:
        curr, depth = stack.pop()

        if depth >= depth_limit:
            continue

        neighbors = adj[curr]
        if len(neighbors) > max_neighbors:         
            neighbors = neighbors[:max_neighbors]

        for nbr in neighbors:
            if nbr in visited:
                score += 5                          # revisit = cycle-like pattern
            else:
                score += 1
                visited.add(nbr)
                stack.append((nbr, depth + 1))

    return score

#flow score
bfs_layering_score = {}
dfs_flow_score = {}
for node in nodes:
    dfs_flow_score[node] = flow_score_iter(node, adj_list)
    bfs_layering_score[node] = bfs_layering(node, adj_list)

accounts = pd.read_csv('elliptic_txs_classes.csv')
account_no = accounts["txId"].astype(int).tolist()
label_map = accounts.set_index("txId")["class"].to_dict()

rows = []
for node in account_no:
    label = label_map.get(node, "unknown")
    if label == "1":
        label = 1
    elif label == "2":
        label = 0
    else:
        label = -1

    rows.append({
        "txId": node,
        "out_degree": degree.get(node, (0,0))[0],
        "in_degree": degree.get(node, (0,0))[1],
        "pagerank": pr.get(node, 0),
        "depth": depth.get(node, 0),
        "clustering": clustering.get(node, 0),
        "chain_score": chain_score.get(node, 0),
        "in_star_score": in_star_score.get(node, 0),
        "out_star_score": out_star_score.get(node, 0),
        "triangle_score": triangle_score.get(node, 0),
        "dfs_flow_score": dfs_flow_score.get(node, 0),
        "bfs_layering_score": bfs_layering_score.get(node, 0),
        "scc_size": scc_size_score.get(node, 0),
        "scc_density": scc_density.get(node, 0),
        "label": label
    })

graph_df = pd.DataFrame(rows)
graph_df.to_csv('graph_heuristics_features.csv', index=False)
print("graph_heuristics_features.csv created")    
