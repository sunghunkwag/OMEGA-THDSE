import ast
import zlib
import networkx as nx
import numpy as np

class TopologicalASTGraphCA:
    def __init__(self, steps: int = 5):
        self.steps = steps

    def get_node_seed(self, node_type: str, evolutionary_pressure: float = 0.0) -> float:
        """
        [MUTATION PROTOCOL INTEGRATION] 
        Deterministically mutate the CA seed if logical deadlock (UNSAT) occurs.
        """
        # Base structural hash
        h = zlib.crc32(node_type.encode('utf-8')) & 0xFFFFFFFF
        base_phase = float(h % 10000) / 10000.0 * 2 * np.pi
        
        # OMEGA FORGE MUTATION INJECTION: 
        # Shift topology deterministically based on evolutionary pressure
        return (base_phase + evolutionary_pressure) % (2 * np.pi)

    def code_to_graph(self, code: str = None, tree: ast.AST = None, evolutionary_pressure: float = 0.0) -> nx.DiGraph:
        if tree is None:
            tree = ast.parse(code)
        
        G = nx.DiGraph()
        
        def traverse(node, parent_id=None, depth=0, sibling_index=0):
            node_id = id(node)
            node_type = type(node).__name__
            lineno = getattr(node, 'lineno', -1)
            
            phase = self.get_node_seed(node_type, evolutionary_pressure)
            G.add_node(
                node_id, 
                type=node_type, 
                state=np.exp(1j * phase),
                lineno=lineno,
                depth=depth,
                sibling_index=sibling_index
            )
            
            if parent_id is not None:
                G.add_edge(parent_id, node_id)
                
            for i, child in enumerate(ast.iter_child_nodes(node)):
                traverse(child, node_id, depth + 1, i)
                
        traverse(tree)
        return G

    def evolve(self, G: nx.DiGraph) -> nx.DiGraph:
        for _ in range(self.steps):
            new_states = {}
            for node in G.nodes:
                state = G.nodes[node]['state']
                
                parents = list(G.predecessors(node))
                children = list(G.successors(node))
                
                parent_msg = sum(G.nodes[p]['state'] for p in parents) if parents else 0
                child_msg = sum(G.nodes[c]['state'] for c in children) if children else 0
                
                agg = state + 0.5 * parent_msg + 0.5 * child_msg
                if np.isclose(np.abs(agg), 0, atol=1e-8):
                    agg = 1.0 + 0j
                
                new_phase = np.angle(agg)
                new_states[node] = np.exp(1j * new_phase)
                
            for node, new_state in new_states.items():
                G.nodes[node]['state'] = new_state
        return G
