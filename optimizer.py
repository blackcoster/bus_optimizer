import sys
sys.path.append('../')
import momepy
import geopandas as gpd
import networkx as nx
import numpy as np
import random
from statistics import mean
from iduedu import get_adj_matrix_gdf_to_gdf
import pygad as pg
import warnings
warnings.filterwarnings("ignore")

def get_walk_bus_car_graph(intermodal_graph):
    G = nx.MultiDiGraph()

    for node,data in intermodal_graph.nodes(data=True):
        data['node_id'] = node

    # Iterate over the edges in the original graph
    for u, v, key, data in intermodal_graph.edges(keys=True, data=True):
        data['edge_id'] = (u,v)
        data['node_start'] = u
        data['node_end'] = v
        data['kee'] = key
        if data['type'] in ['walk', 'bus','car']:
            G.add_edge(u, v, key=key, **data)

    # Add the corresponding nodes to the subgraph
    for node in intermodal_graph.nodes:
        if G.has_node(node):  # Only add nodes that are connected by walk or bus edges
            G.nodes[node].update(intermodal_graph.nodes[node])
    G.graph = intermodal_graph.graph
    return G

def separate_graphs(walk_bus_car_graph):
    # creating walk graph
    walk = nx.MultiDiGraph()
    # Iterate over the edges in the original graph
    for u, v, key, data in walk_bus_car_graph.edges(keys=True, data=True):
        if data['type'] in ['walk']:
            data['time_min'] = data['length_meter']/84
            walk.add_edge(u, v, key=key, **data)
    # Add the corresponding nodes to the subgraph
    for node in walk_bus_car_graph.nodes:
        if walk.has_node(node):  # Only add nodes that are connected by walk or bus edges
            walk.nodes[node].update(walk_bus_car_graph.nodes[node])
    walk.graph = walk_bus_car_graph.graph


    # creating car graph
    car_g = nx.MultiDiGraph()
    # Iterate over the edges in the original graph
    for u, v, key, data in walk_bus_car_graph.edges(keys=True, data=True):
        if data['type'] in ['car']:
            data['time_min'] = data['length_meter']/1000
            car_g.add_edge(u, v, key=key, **data)
    # Add the corresponding nodes to the subgraph
    for node in walk_bus_car_graph.nodes:
        if car_g.has_node(node):  # Only add nodes that are connected by walk or bus edges
            car_g.nodes[node].update(walk_bus_car_graph.nodes[node])
    car_g.graph = walk_bus_car_graph.graph


    # creating bus graph
    bus_g = nx.MultiDiGraph()
    # Iterate over the edges in the original graph
    for u, v, key, data in walk_bus_car_graph.edges(keys=True, data=True):
        if data['type'] in ['bus']:
            data['time_min'] = data['length_meter']/1000
            bus_g.add_edge(u, v, key=key, **data)
    # Add the corresponding nodes to the subgraph
    for node in walk_bus_car_graph.nodes:
        if bus_g.has_node(node):  # Only add nodes that are connected by walk or bus edges
            bus_g.nodes[node].update(walk_bus_car_graph.nodes[node])
    bus_g.graph = walk_bus_car_graph.graph

    return walk, car_g, bus_g

def extract_routes(bus_e):
    routes = dict()
    for i in set(bus_e['desc']):
        routes[int(i.split()[1])] = bus_e[bus_e['desc']==i]
    nums = sorted(list(set(bus_e['desc'])))
    bus_nums = sorted([int(i.split()[1]) for i in nums]) # list of bus numbers

    # each route - a list of tuples of edge pairs
    routes_list = dict()
    for i in bus_nums:
        start_node = [start for start,end in routes[i]['edge_id'] if start not in [end for start,end in routes[i]['edge_id']]][0]

        # Create a dictionary to map node_start to node_end
        path_dict  = {start: end for start, end in routes[i]['edge_id']}

        node_start = [edge[0] for edge in routes[i]['edge_id']]
        node_end = [edge[1] for edge in routes[i]['edge_id']]

        ordered_edges = []
        an = [node_start]
        current_node = start_node

        while current_node in path_dict:
            next_node = path_dict[current_node]
            ordered_edges.append((current_node, next_node))
            current_node = next_node
            an.append(current_node)
        routes_list[i] = ordered_edges
    return routes_list
    
def encode_routes(routes_list):
    # creates a one big chromosome of all routes together (long) and a splits dict to extract them by indexes from it later
    long = []
    splits = {}
    start=0
    for k in routes_list.keys():
        long.extend(routes_list[k])
        splits[k] = (start,start+len(routes_list[k]))
        start+=len(routes_list[k])
    return long, splits


from shapely.geometry import LineString
def split_line_at_point(line, point):
    coords = list(line.coords)
    for i in range(len(coords) - 1):
        segment = LineString([coords[i], coords[i + 1]])
        if segment.distance(point) < 0.0000001:
            return (LineString(coords[:i + 1] + [(point.x, point.y)]),
                    LineString([(point.x, point.y)] + coords[i + 1:]))


def add_node_to_graph(graph, point,uds):
    
    car_e = momepy.nx_to_gdf(uds)[1]
    nearest_edge_candidates = gpd.sjoin_nearest(point.reset_index(), car_e, how="left")
    edge_key = nearest_edge_candidates.iloc[0]['kee']
    nearest_edge = car_e[(car_e['edge_id'] == nearest_edge_candidates.iloc[0]['edge_id']) & (car_e['kee'] == edge_key)]
    nearest_point = nearest_edge.reset_index().geometry.interpolate(nearest_edge.reset_index().geometry.project(point.reset_index().geometry))

    distance_to_nearest_edge = nearest_edge.reset_index().geometry.distance(point.reset_index().geometry.iloc[0]).iloc[0]

    def getkey(u,v):
        if graph.has_edge(u, v):
            keys = list(graph[u][v].keys())
        else:
            keys = []
        
        if keys:
            max_key = max(keys)
        else:
            max_key = -1

        new_key = max_key + 1
        return new_key

    if distance_to_nearest_edge < 3:
        new_node_id = int(max(nx.get_node_attributes(graph, 'node_id').values()) + 1)
        
        # Add the new node to the graph
        graph.add_node(new_node_id, 
                    x=float(nearest_point.x[0]), 
                    y= float(nearest_point.y[0]), 
                    node_id = new_node_id,
                    geometry=nearest_point.geometry[0],
                    stop=True,
                    desc='proj'
                    )
        u, v = nearest_edge.reset_index().iloc[0]['edge_id']
        if graph.has_edge(u, v):
            graph.remove_edge(u, v)
        
        line1 = split_line_at_point(nearest_edge.reset_index().geometry[0],nearest_point[0])[0]
        line2 = split_line_at_point(nearest_edge.reset_index().geometry[0],nearest_point[0])[1]
        # Add edges to the new node
        graph.add_edge(int(u), new_node_id,
                    length_meter=line1.length, 
                    geometry=line1, 
                    type='car',
                    time_min=line1.length/1000,
                    edge_id= (int(u),new_node_id),
                    desc='',
                    kee = getkey(int(u), new_node_id),
                    node_start = int(u),
                    node_end=new_node_id
                    )
        graph.add_edge(new_node_id, int(v), 
                    length_meter=line2.length, 
                    geometry=line2, 
                    type='car',
                    time_min=line2.length/1000,
                    edge_id= (new_node_id,int(v)),
                    desc='',
                    kee = getkey(new_node_id, int(v)),
                    node_start=new_node_id,
                    node_end = int(v)
                    )  
        initial_point_geom = point.reset_index().geometry.iloc[0]
        projection_geom = nearest_point
    
        # Create a new edge id
        graph.add_edge(int(point.reset_index()['node_id'][0]), new_node_id, 
                    length_meter=float(0.0), 
                    geometry=LineString([(initial_point_geom.x, initial_point_geom.y), (projection_geom.x, projection_geom.y)]),
                    type='car',
                    time_min=float(0),
                    edge_id=(int(point.reset_index()['node_id'][0]), new_node_id))
        
        graph.add_edge( new_node_id, int(point.reset_index()['node_id'][0]),
                    length_meter=float(0.0), 
                    geometry=LineString([(projection_geom.x, projection_geom.y), (initial_point_geom.x, initial_point_geom.y)]),
                    type='car',
                    time_min=float(0),
                    edge_id=(new_node_id, int(point.reset_index()['node_id'][0])))

    return graph



def create_subgraph(graph, edge_types):
    # Create a subgraph that will contain only the specified edge types
    subgraph = nx.MultiDiGraph()
    
    for u, v, key, data in graph.edges(keys=True, data=True):
        if data.get('type') in edge_types:
            data['time_min'] = data['length_meter']/1000
            subgraph.add_edge(u, v, key=key, **data)
    
    # Add nodes with their attributes to the subgraph
    for node in subgraph.nodes():
        subgraph.nodes[node].update(graph.nodes[node])
    subgraph.graph = graph.graph
    return subgraph


# find out which bus nodes could become intermediate stops for existing bus edges
# returns a list of possible connections [(a,b), (a,x,b), (a,y,b)]
def find_reachable_bus_nodes(carbus, edge, bus_g, max_length=2000):
    e1,e2 = edge
    
    shortest_paths_from_e1 = nx.single_source_dijkstra_path_length(carbus, e1, weight='length_meter')
    shortest_paths_to_e2 = nx.single_source_dijkstra_path_length(carbus.reverse(), e2, weight='length_meter')

    # genes = [([edge,carbus[e1][e2][0]['length_meter']])]
    genes = [edge]

    reachable_bus_nodes = [
        node for node in bus_g.nodes 
        if node in shortest_paths_from_e1 and node in shortest_paths_to_e2 and node not in edge and
        shortest_paths_from_e1[node] + shortest_paths_to_e2[node] < max_length 
    ]
    for node in reachable_bus_nodes:
        # genes.append(((e1,node,e2),(shortest_paths_from_e1[node], shortest_paths_to_e2[node])))
        genes.append((e1,node,e2))
    
    return genes


# makes a single little different individual/
# from initial by randomly choosing an available option for all the genes/
# available chices are stored in gene_variations
def create_individual(orig_gene, gene_variations):
    individual = []
    for gene in orig_gene:
        variation = random.choice(gene_variations[gene])
        individual.append(variation)
    return individual

#just make many individuals by pop_size. list
def initialize_population(orig_gene, gene_variations, population_size):
    population = []
    for _ in range(population_size):
        individual = create_individual(orig_gene, gene_variations)
        population.append(individual)
    return population


def possible_modifications(carbus,routes_list,bus_g):
    # compar= gene_variations. in each route for each gene outlines possible modifications
    compar = dict()
    for k in routes_list.keys():
        for edge in routes_list[k]:
            compar[edge]= find_reachable_bus_nodes(carbus,edge,bus_g)
    return compar

# extracts from carbus the data for the freshly created edges (using drive)
from shapely.ops import linemerge

def get_geom_from_edge(carbus, edge,route):
    u, v = edge
    
    # Calculate the shortest path using Dijkstra's algorithm
    path = nx.dijkstra_path(carbus, u, v, weight='time_min')
    
    geometries = []
    # Extract the geometry and other attributes of each edge along the path
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edges = carbus.get_edge_data(u, v)
        # Find the edge with the minimum time_min
        fastest_edge_key = min(edges, key=lambda k: edges[k]['time_min'])
        edge_data = edges[fastest_edge_key]
        geometries.append(edge_data['geometry'])
    u, v = edge
    u_x, u_y = carbus.nodes[u]['x'], carbus.nodes[u]['y']
    v_x, v_y = carbus.nodes[v]['x'], carbus.nodes[v]['y']
    direct_line = LineString([(u_x, u_y), (v_x, v_y)])
    combined_geometry = linemerge(geometries)
    combined_edge_data = {
        'edge_id': edge,
        'geometry': direct_line,
        'length_meter': combined_geometry.length,
        'time_min': combined_geometry.length/1000,
        'type': 'bus',
        'desc': f'route {route}'
    }
    return combined_edge_data

# unpacking the chromosome into a full graph (putting new bus system into walk)
def chrom_to_graph(chrom_long,carbus,splits,walk):
    new = walk.copy()
    for key,value in splits.items():
        a,b = value
        chrom = chrom_long[a:b]
        for gene in chrom:
            edges=gene
            if len(edges)==2:
                new.add_edge(edges[0],edges[1],**get_geom_from_edge(carbus, edges,key))
            if len(edges)==3:
                new.add_edge(edges[0],edges[1],**get_geom_from_edge(carbus, edges[0:2],key))
                new.add_edge(edges[1],edges[2],**get_geom_from_edge(carbus, edges[1:],key))
            
    return new


# getweighted connectivity both ways. Routes from home to service get more weight
# as they are more popular (evaluated by A poplutaion * B num of services * time(A,B))
# Backwards connectivity is evaluated by A num of services * B population * time(A,B)). A float for each
def home_ser(adj_mx,gdf):
  data = []
  for i in adj_mx.index:
    pi = gdf.loc[i, 'population']
    for j in adj_mx.columns:
      sj = gdf.loc[j, 'services']
      dij = adj_mx.loc[i,j]
      conn_ij = 1/(dij if dij>1 else 1)
      data.append(pi*sj*conn_ij)

  conn_ps = mean(data) # -> max
  return conn_ps

def ser_home(adj_mx,gdf):
  data = []
  for i in adj_mx.index:
    si = gdf.loc[i, 'services']
    for j in adj_mx.columns:
      pj = gdf.loc[j, 'population']
      dij = adj_mx.loc[i,j]
      conn_ij = 1/(dij if dij>1 else 1)
      data.append(si*pj*conn_ij)

  conn_ps = mean(data) # -> max
  return conn_ps


def genetic(carbus, splits,walk,blocks, id_to_path,gene_space_encoded,num_generations=25):
    # finess function evaluation. 2 weighted connectivities and length increase
    def func(ga_instance, solution, solution_idx):
        solution = [id_to_path[gene] for gene in solution]
        graph = chrom_to_graph(solution,carbus,splits,walk)
        e = momepy.nx_to_gdf(graph)[1]
        bus = e[e['type']=='bus']
        notorig = bus.groupby('desc')['length_meter'].sum().reset_index()
        for e1,e2,data in graph.edges(data=True):
            data['weight'] = data['time_min']
            data['transport_type'] = data['type']

        # ac = AdjacencyCalculator(blocks=vasya, graph=graph)
        # adj_mx = ac.get_dataframe()
        adj_mx = get_adj_matrix_gdf_to_gdf(blocks,blocks,graph,'weight',np.float64)
        ps = home_ser(adj_mx,blocks)
        sp = ser_home(adj_mx,blocks)
        
        # difference= diff(orig,notorig)
        difference = 1/notorig['length_meter'].sum()

        return ps, sp, difference
    
    
    ga = pg.GA(
    num_generations=num_generations,
    num_parents_mating=2,
    fitness_func=func,
    gene_space=gene_space_encoded,
    sol_per_pop=10,
    num_genes=len(gene_space_encoded)
    )
    ga.run()
    best_solution, best_fit, _ = ga.best_solution()
    decoded = [id_to_path[o] for o in best_solution]
    graph = chrom_to_graph(decoded,carbus,splits,walk)
    return graph, best_fit
    
def int_gene_space(gene_space):
    all_paths = set()
    for options in gene_space:
        all_paths.update(options)
    path_to_id = {p: i for i, p in enumerate(all_paths)}
    id_to_path = {i: p for p, i in path_to_id.items()}

    gene_space_encoded = [[path_to_id[path] for path in elem] for elem in gene_space]

    return path_to_id, id_to_path, gene_space_encoded