"""."""
import random
from pickle import load

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pygame


class Body:
    """."""
    def __init__(self, pos_, mass):
        self.pos = pos_
        self.mass = mass
        self.force = np.zeros(2, dtype=np.float64)
        self.vel = np.zeros(2, dtype=np.float64)

    def __repr__(self):
        return f"Body({self.pos}, {self.mass})"

    def update_pos(self,dt_):
        """."""
        self.vel = (self.vel + (self.force/self.mass)*dt_)*DAMP
        # speed_sq = self.vel[0]**2 + self.vel[1]**2
        # if speed_sq > MAX_SPEED_SQ:
        #     self.vel = (self.vel / np.sqrt(speed_sq)) * MAX_SPEED
        self.vel[0] = min(max(self.vel[0],-MAX_SPEED),MAX_SPEED)
        self.vel[1] = min(max(self.vel[1],-MAX_SPEED),MAX_SPEED)
        self.pos += self.vel*dt_

    def clear_forces(self):
        """."""
        self.force = np.zeros(2, dtype=np.float64)


class Node:
    """."""
    def __init__(self, center, half_width, half_height):
        self.half_width = half_width
        self.half_height = half_height
        self.test_sq = min(half_width*2,half_height*2)**2
        self.center = center
        self.children = None
        self.body = None
        self.center_of_mass = np.zeros(2, dtype=np.float64)
        self.mass = 0.0
        self.num_bodies = 0

    def has_children(self):
        """."""
        return self.children is not None

    def is_empty(self):
        """."""
        return self.body is None and self.children is None

    def is_leaf(self):
        """."""
        return self.body is not None

    def get_quad(self, pos_):
        """."""
        return int(pos_[0] > self.center[0]) | (int(pos_[1] > self.center[1]) << 1)

    def make_children(self):
        """."""
        hhw = self.half_width / 2
        hhh = self.half_height / 2
        cx, cy = self.center
        self.children = [
            Node((cx - hhw, cy - hhh), hhw, hhh),  # bottom-left
            Node((cx + hhw, cy - hhh), hhw, hhh),  # bottom-right
            Node((cx - hhw, cy + hhh), hhw, hhh),  # top-left
            Node((cx + hhw, cy + hhh), hhw, hhh)   # top-right
        ]
        body_ = self.body
        self.body = None
        return body_

    def compute_com(self):
        """Calculate COM for the entire subtree in one pass"""
        if self.is_empty() and not self.has_children():
            self.mass = 0.0
            self.center_of_mass = np.zeros(2, dtype=np.float64)
            return

        if not self.has_children():
            self.center_of_mass = self.body.pos.copy()
            self.mass = self.body.mass
            self.num_bodies = 1
            return

        self.mass = 0.0
        self.center_of_mass = np.zeros(2, dtype=np.float64)

        for child in self.children:
            child.compute_com()
            if child.mass > 0:
                self.mass += child.mass
                self.center_of_mass += child.center_of_mass * child.mass
                self.num_bodies += child.num_bodies

        if self.mass > 0:
            self.center_of_mass /= self.mass

class Quadtree:
    """."""
    def __init__(self, center, half_width, half_height):
        self.root = Node(center, half_width, half_height)

    def add(self, body_):
        """Non-recursive implementation to add a body"""
        node_ = self.root

        while True:

            if not node_.has_children():
                if node_.is_empty():
                    # Empty leaf node, add body here
                    node_.body = body_
                    return
                else:
                    # Need to split this node
                    old_body = node_.make_children()  # Create children and get the old body
                    # Don't add old_body to update_path since we'll handle it now

                    # Insert the old body into appropriate child
                    quad = node_.get_quad(old_body.pos)
                    child = node_.children[quad]
                    child.body = old_body
                    child.center_of_mass = old_body.pos.copy()
                    child.mass = old_body.mass

                    # Continue with current body
                    quad = node_.get_quad(body_.pos)
                    node_ = node_.children[quad]
                    # No need to add to update_path since we already updated it
            else:
                # Internal node with children, descend to appropriate child
                quad = node_.get_quad(body_.pos)
                node_ = node_.children[quad]

    def build_tree(self, bodies_):
        """Build the tree from a collection of bodies in one go"""
        for body_ in bodies_:
            self.add(body_)

    def compute_all_coms(self):
        """Compute all centers of mass in one bottom-up pass"""
        self.root.compute_com()

    def set_forces(self,bodies_:list[Body],theta=0.5, epsilon = 1e-5):
        """."""
        t_sq = theta**2
        e_sq = epsilon**2
        for body_ in bodies_:
            stack_ = [self.root]
            while stack_:
                node_ = stack_.pop()
                if node_.is_empty():
                    continue
                diff = body_.pos - node_.center_of_mass
                dist_sq = diff[0]**2 + diff[1]**2 + e_sq
                if node_.is_leaf() or (node_.test_sq / dist_sq) < t_sq:
                    # approximation satisfied, compute force
                    inv_r2 = 1/dist_sq
                    inv_r = np.sqrt(inv_r2)
                    inv_r3 = inv_r2*inv_r
                    body_.force += REPULSE * diff * inv_r3 * node_.num_bodies
                else:
                    # approximation invalid, recurse
                    stack_.extend(node_.children)


# Example usage with optimized batch processing
def add_repulsive_forces(bodies_:list[Body]):
    """."""
    xs = [b.pos[0] for b in bodies_]
    ys = [b.pos[1] for b in bodies_]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    hw = (max_x - min_x)/2
    hh = (max_y - min_y)/2
    # Build tree first, then compute COM in one pass
    q_tree = Quadtree((hw+min_x, hh+min_y), hw,hh)
    q_tree.build_tree(bodies_)
    q_tree.compute_all_coms()
    # compute forces on each body and update position
    q_tree.set_forces(bodies_, theta=THETA)
    return q_tree


WIDTH =1800
HEIGHT = 1000
SPAWN_OFFSET = 250
NODE_RADIUS = 4
SPRING = 2
REPULSE = 15000
DT = 0.2
MAX_SPEED = 50
MAX_SPEED_SQ = MAX_SPEED**2
DAMP = 0.75
THETA = 0.9

EDGE_REPEL = True
NODE_REPEL = True
SHOW_QTREE = False

if __name__ == "__main__":
    # --- Init Graph ---
    with open('network_edges_lite.pickle','rb') as f:
        EDGES = load(f)

    edges = []
    nodes = set()
    for edge in sorted(EDGES,key=lambda e: abs(e[2]["weight"]), reverse=True)[:1000]:
        n1, n2, properties = edge
        weight = properties["weight"]
        color = properties["color"]
        nodes.add(n1)
        nodes.add(n2)
        edges.append(edge)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # isolate connected components
    for c in sorted(nx.connected_components(G), key=len, reverse=True):
        if len(c) <= 5:
            continue
        S = G.subgraph(c).copy()
        edge_colors = [S[u][v]['color'] for u, v in S.edges()]
        bodies = {n:Body(
            np.array([random.randint(SPAWN_OFFSET, WIDTH-SPAWN_OFFSET)+random.random(),
                       random.randint(SPAWN_OFFSET, HEIGHT-SPAWN_OFFSET)+random.random()],
                      dtype=np.float64), S.degree[n] + 1) for n in S.nodes}
        edge_bodies = {(u,v):Body(bodies[v].pos + (bodies[u].pos - bodies[v].pos)/2, 1)
                       for u,v in S.edges()}

        # --- Init Pygame ---
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT),flags=0)
        clock = pygame.time.Clock()

        RUNNING = True

        # --- Main Loop ---
        while RUNNING:
            screen.fill((30, 30, 30)) #dark background

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    RUNNING = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        DT += 0.01
                    if event.key == pygame.K_DOWN:
                        DT -= 0.01
                    if event.key == pygame.K_LEFT:
                        REPULSE -= 1000
                    if event.key == pygame.K_RIGHT:
                        REPULSE += 1000
                    if event.key == pygame.K_e:
                        EDGE_REPEL = not EDGE_REPEL
                    if event.key == pygame.K_n:
                        NODE_REPEL = not NODE_REPEL
                    if event.key == pygame.K_q:
                        SHOW_QTREE = not SHOW_QTREE

            #reset forces
            for body in bodies.values():
                body.clear_forces()
            for body in edge_bodies.values():
                body.clear_forces()

            # Compute repulsive forces
            if NODE_REPEL:
                q_n = add_repulsive_forces(bodies.values())

            if EDGE_REPEL:
                q_e = add_repulsive_forces(edge_bodies.values())
                for u,v in S.edges():
                    bodies[u].force += edge_bodies[(u,v)].force
                    bodies[v].force += edge_bodies[(u,v)].force

            # Compute spring (attractive) forces
            for u, v in S.edges:
                delta = bodies[u].pos - bodies[v].pos
                dist = np.linalg.norm(delta) + 0.01
                spring_force = -SPRING * dist
                direction = delta / dist
                bodies[u].force += direction * spring_force
                bodies[v].force -= direction * spring_force

            # Update velocities and positions
            for n in S.nodes:
                bodies[n].update_pos(DT)
                # Keep nodes within bounds
                bodies[n].pos = np.clip(bodies[n].pos, NODE_RADIUS,
                                        [WIDTH - NODE_RADIUS, HEIGHT - NODE_RADIUS])

            for u, v in S.edges():
                edge_bodies[(u,v)].pos = bodies[v].pos + (bodies[u].pos - bodies[v].pos)/2


            # Draw edges
            for u, v in S.edges:
                pygame.draw.line(screen, (200, 200, 200), bodies[u].pos, bodies[v].pos, 1)
                pygame.draw.circle(screen, (200,100,100), edge_bodies[(u,v)].pos, 2)

            # Draw nodes
            for n in S.nodes:
                pygame.draw.circle(screen, (100, 200, 250), bodies[n].pos, NODE_RADIUS)


            #draw quad tree
            if SHOW_QTREE:
                if NODE_REPEL:
                    stack = [q_n.root]
                    while stack:
                        node = stack.pop()
                        if node.is_empty() or node.is_leaf():
                            continue
                        pygame.draw.line(screen,
                                         (0,0,250),
                                         (node.center[0] - node.half_width, node.center[1]),
                                         (node.center[0] + node.half_width, node.center[1]),
                                         1)
                        pygame.draw.line(screen,
                                         (0,0,250),
                                         (node.center[0],node.center[1] - node.half_height),
                                         (node.center[0],node.center[1] + node.half_height),
                                         1)
                        if node.has_children():
                            stack.extend(node.children)
                if EDGE_REPEL:
                    stack = [q_e.root]
                    while stack:
                        node = stack.pop()
                        if node.is_empty() or node.is_leaf():
                            continue
                        pygame.draw.line(screen,
                                         (250,0,0),
                                         (node.center[0] - node.half_width, node.center[1]),
                                         (node.center[0] + node.half_width, node.center[1]),
                                         1)
                        pygame.draw.line(screen,
                                         (250,0,0),
                                         (node.center[0],node.center[1] - node.half_height),
                                         (node.center[0],node.center[1] + node.half_height),
                                         1)
                        if node.has_children():
                            stack.extend(node.children)


            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

        plt.figure(figsize=(10.8,19.2))
        pos = {n: bodies[n].pos*2 for n in S.nodes}
        nx.draw_networkx_nodes(S,pos, node_color='lightblue')
        nx.draw_networkx_labels(S,pos, font_size=8)
        nx.draw_networkx_edges(S,pos,edge_color=edge_colors)
        # nx.draw_networkx_edge_labels(S,pos,edge_labels=nx.get_edge_attributes(S,'weight'))
        plt.show()
