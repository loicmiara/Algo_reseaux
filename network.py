import geopandas as gpd
import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
import os
import folium
from folium.plugins import Fullscreen
import alphashape
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.animation as animation
import imageio
import networkx as nx
import io
from PIL import Image
import time
import sys
from matplotlib import cm
from IPython.display import clear_output,display
import seaborn as sns
import matplotlib.colors as mcolors
from shapely import box
import gc
pd.options.mode.copy_on_write = True
import scipy
from shapely.geometry import Point
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from ipywidgets import interact, FloatSlider
from rasterstats import point_query
from datetime import datetime
import shutil
from shortest_path_util import penalty_turns
from shortest_path_turn_penalty import shortest_path_turn_penalty

def is_almost_subset(a, b, tolerance=0.1):
    if not a:
        return True
    common = a & b
    match_fraction = len(common) / len(a)
    return match_fraction >= (1 - tolerance)

def get_mask(x, all_routes, tolerance):
    return [
        (route != x) and (len(route) != 0) and is_almost_subset(route, x, tolerance=tolerance)
        for route in all_routes
    ]

def count_left_turns(edges):
    angles = np.diff(edges)%360
    n_left_turns = (angles<330)*(angles>150)
    return n_left_turns.sum()

def count_right_turns(edges):
    angles = np.diff(edges)%360
    n_right_turns = (angles<=150)*(angles>30)
    return n_right_turns.sum()

def split_components(gdf,cycleway_nodes):

    gdf = gdf.copy()
    df = gdf.reset_index()  # in case index is (u,v,key)

    prev_u = df['u'].shift()
    prev_v = df['v'].shift()
    
    connected_to_prev = (
        (df['u'] == prev_u) |
        (df['u'] == prev_v) |
        (df['v'] == prev_u) |
        (df['v'] == prev_v)
    )

    df['component'] = (~connected_to_prev).cumsum() - 1
    gdf['component']=df['component'].values
    
    g = gdf.reset_index().groupby('component')
    component_lengths = g['length'].sum()
    gdf['component_length'] = gdf['component'].map(component_lengths)
    first_u = g['u'].first()
    last_v  = g['v'].last()
    start_conn = first_u.isin(cycleway_nodes)
    end_conn   = last_v.isin(cycleway_nodes)
    gdf['start_connected'] = gdf['component'].map(start_conn)
    gdf['end_connected']   = gdf['component'].map(end_conn)
    gdf['both_connected'] = gdf['start_connected'] & gdf['end_connected']

    return gdf
    
def group_street_components(gdf, name_col="street_name"):
    # Result container
    all_results = []

    # Process each street name separately
    for name, group in gdf.groupby(name_col):
        group = group.copy()
        group = group.reset_index(drop=False)  # keep original index

        # Build graph
        G = nx.Graph()
        G.add_nodes_from(group.index)

        # Spatial index speeds up the touching lookup
        sindex = group.sindex

        for i, geom in group.geometry.items():
            # Candidate neighbors using bounding box intersection
            possible = list(sindex.intersection(geom.bounds))
            possible.remove(i)  # remove itself

            for j in possible:
                if geom.touches(group.geometry[j]) or geom.intersects(group.geometry[j]):
                    G.add_edge(i, j)

        # Extract connected components
        components = list(nx.connected_components(G))

        # Assign component labels
        comp_map = {}
        for comp_id, comp_nodes in enumerate(components):
            for n in comp_nodes:
                comp_map[n] = comp_id

        group["component_id"] = group.index.map(comp_map)
        group["project"] = (
            group[name_col].astype(str) + "_" + group["component_id"].astype(str)
        )

        all_results.append(group)

    return gpd.GeoDataFrame(pd.concat(all_results).set_index(['u','v','key']))
    
class Network:
    def __init__(self, crs = 32618):
        self.crs = crs
        self.n_pot = None
        self.n_pot_og = None
        self.n_pot_nodes = None
        self.n_pot_edges = None
        self.n_ex = None
        self.n_ex_og = None
        self.n_ex_nodes = None
        self.n_ex_edges = None
        self.trips = None
        self.map_pot = None
        self.map_ex = None
        self.map_routes = None
        self.trips = None
        self.trips_within = None
        self.sample = None
        self.boundaries = None
        self.routes = None
        self.unsolved_routes_ids = []
        self.all_routes_edges = None
        self.routes_summary = None
        self.evol = []
        self.associated_links = None
        self.transit_stops = None
        self.traffic_signals = None
        self.turn_penalties = None
        self.iterations_description = pd.DataFrame(columns=['generalized_benefit','added_length','n_near_completed_routes','n_recomputed_routes','n_changed_routes'])
        self.cycleway_nodes = None
        self.projects = None
        
        
    def load_n_pot(self,path):
        self.n_pot = ox.io.load_graphml(path)
        self.n_pot = ox.project_graph(self.n_pot)
        self.n_pot_nodes,self.n_pot_edges = ox.graph_to_gdfs(self.n_pot)
        used_nodes = set(self.n_pot_edges.index.get_level_values('u')) | set(self.n_pot_edges.index.get_level_values('v'))
        self.n_pot_nodes = self.n_pot_nodes[self.n_pot_nodes.index.isin(used_nodes)]
        self.n_pot_edges['build_iter'] = 0
        self.n_pot_og = ox.graph_from_gdfs(self.n_pot_nodes.copy(),self.n_pot_edges.copy())
        
    def filter_n_pot(self):
        self.n_pot_edges[self.n_pot_eges.highway.isin(['residential','primary','secondary','tertiary','tertiary_link'])]
        self.n_pot = ox.graph_from_gdfs(self.n_pot_nodes,self.n_pot_edges)

    def plot_n_pot(self, nodes = False):
        if self.n_pot == None:
            print('No potential network loaded')
        else:
            self.map_pot = self.n_pot_edges.explore(color = 'black')
            if nodes:
                self.n_pot_nodes.explore(m = self.map_pot)
            display(self.map_pot)

    def load_n_ex(self,path):
        self.n_ex = ox.io.load_graphml(path)
        self.n_ex = ox.project_graph(self.n_ex)
        self.n_ex_nodes,self.n_ex_edges = ox.graph_to_gdfs(self.n_ex)
        used_nodes = set(self.n_ex_edges.index.get_level_values('u')) | set(self.n_ex_edges.index.get_level_values('v'))
        self.n_ex_nodes = self.n_ex_nodes[self.n_ex_nodes.index.isin(used_nodes)]
        self.n_ex_edges['build_iter'] = 0
        self.n_ex_og = ox.graph_from_gdfs(self.n_ex_nodes.copy(),self.n_ex_edges.copy())

    def filter_n_ex(self,highway_type = 'cycleway'):
        self.n_ex_edges = self.n_ex_edges[self.n_ex_edges.highway == highway_type]
        self.n_ex = ox.graph_from_gdfs(self.n_ex_nodes,self.n_ex_edges)

    def plot_n_ex(self, nodes = False):
        if self.n_ex == None:
            print('No existing network loaded')
        else:
            self.map_ex = self.n_ex_edges.explore(color = 'black')
            if nodes:
                self.n_ex_nodes.explore(m = self.map_ex)
            display(self.map_ex)

    def load_trips(self, path, source_crs, delimiter = ',',cols = ['ipere','mode','motif','age','xdomi','ydomi','faclog_s27','xorig','yorig','xdest','ydest','potVelo','fexpPotVelo', 'velo'],potvel = True):
        self.trips = pd.read_csv(path,delimiter = delimiter)
        if potvel == True:
            self.trips = self.trips[self.trips.potVelo == 1]
        else:
            self.trips = self.trips[(self.trips.potVelo == 1)|(self.trips.velo == 'X')]
        self.trips = self.trips[cols]
        self.trips = gpd.GeoDataFrame(self.trips, geometry=gpd.points_from_xy(self.trips.xorig, self.trips.yorig), crs=source_crs)
        self.trips = self.trips.rename(columns = {'geometry': 'orig'})
        self.trips = self.trips.assign(dest = gpd.points_from_xy(self.trips.xdest, self.trips.ydest,crs=source_crs))
        # self.trips = self.trips.assign(domi = gpd.points_from_xy(self.trips.xdomi, self.trips.ydomi,crs=source_crs))
        self.trips = self.trips.set_geometry('orig')
        self.trips = self.trips.set_crs(source_crs)
        self.trips = self.trips.to_crs(self.crs)
        self.boundaries = gpd.GeoDataFrame({'name':['region']},geometry = gpd.GeoSeries(self.n_pot_edges.geometry.union_all()).concave_hull(0.2)
                                           , crs = self.crs)
        
        self.trips = self.trips.to_crs(self.crs)
        self.trips.index+=1
        self.trips_within = gpd.sjoin(self.trips, self.boundaries, how='inner', predicate='within')[list(self.trips.columns)]
        self.trips_within = self.trips_within.set_geometry('dest')
        self.trips_within = self.trips_within.to_crs(self.crs)
        self.trips_within = gpd.sjoin(self.trips_within, self.boundaries, how='inner', predicate='within')[list(self.trips.columns)]
        


    def sample_trips(self,sample_size = None, spec_route = None):
        if sample_size is None:
            sample_size = len(self.trips_within)
            self.sample = self.trips_within.sample(sample_size)
        else:
            self.sample = pd.concat([self.sample,self.trips_within.sample(sample_size)])
        if spec_route is not None:
            self.sample = self.trips_within[self.trips_within.ipere == spec_route]

    
    def reset_sample(self):
        self.sample = None
    def reset_routes(self):
        self.routes = []
        self.unsolved_routes_ids = []
        self.all_routes_edges = None
        self.routes_summary = None
        
    def get_routes_edges(self,network):
        route_edges = []
        static_route_ids = []
        for i in tqdm(range(len(self.routes)),leave = False):
            if self.routes.iloc[i].nodes is not None:
                if len(self.routes.iloc[i].nodes)>1:
                    edges = ox.routing.route_to_gdf(network,self.routes.iloc[i]['nodes'])
                    edges['route_number'] = self.sample.index[i]
                    route_edges.append(edges) 
                else:
                    static_route_ids.append(self.sample.index[i])
            
            else:
                self.unsolved_routes_ids.append(self.sample.index[i])
                continue
        print((1-len(self.unsolved_routes_ids)/len(self.sample))*100,'% of solved routes, ',len(static_route_ids)/len(self.sample)*100,
        ' % of static routes')
        self.sample = self.sample.drop(self.unsolved_routes_ids)
        self.sample = self.sample.drop(static_route_ids)
        self.all_routes_edges = pd.concat(route_edges)
    
    
    def compute_routes_summary(self, beta_age,mu_age,scale_age,beta_transit,beta_bikeway,alpha,weight = None, proximity_dist = 10, subsample_idxs = None):
    
        if subsample_idxs is None:
            subsample = self.all_routes_edges
            subsample_idxs = self.sample.index
            self.routes_summary = subsample[['length','gencost','route_number']].groupby(['route_number']).sum()
            self.routes_summary = pd.merge(self.routes_summary,self.sample,left_index=True,right_index=True,how='left')
            if self.projects is not None:
                self.projects_summary = self.projects[['length','project']].groupby('project').sum()
        else:
            subsample = self.all_routes_edges[self.all_routes_edges.route_number.isin(subsample_idxs)]
    
        subsample['flow'] = subsample.join(subsample.index.value_counts(), how = 'left')['count']
        grouped_route_links = subsample.groupby('route_number')
    
        
        
        self.routes_summary.loc[subsample_idxs,'length_cycleway'] = grouped_route_links.apply(
            lambda x: x.loc[x["highway"] == "cycleway", "length"].sum(),include_groups=False)
        self.routes_summary.loc[subsample_idxs,'length_street'] =grouped_route_links.apply(
            lambda x: x.loc[x["highway"] != "cycleway", "length"].sum(),include_groups=False)
        self.routes_summary.loc[subsample_idxs,'total_length'] = self.routes_summary.loc[subsample_idxs,'length_cycleway']+self.routes_summary.loc[subsample_idxs,'length_street'] 
        self.routes_summary.loc[subsample_idxs,'links_id_to_complete'] = grouped_route_links.apply(
            lambda x: set(x.loc[x["highway"] != "cycleway"].index.tolist()), include_groups = False)
        self.routes_summary.loc[subsample_idxs,'prop_cycleway'] = (self.routes_summary.loc[subsample_idxs,'length_cycleway']/
                                                                    (self.routes_summary.loc[subsample_idxs,'length_street']+
                                                                    self.routes_summary.loc[subsample_idxs,'length_cycleway']))
    
        
    
        if self.projects is not None:
            grouped_projects = self.projects.groupby('project')
            self.projects_summary.loc[:,'length_street'] = grouped_projects.apply(
                lambda x: x.loc[x["highway"] != "cycleway", "length"].sum(),include_groups=False)
            
            self.projects_summary.loc[:,'links_id_to_complete'] = grouped_projects.apply(
                lambda x: set(x.loc[x["highway"] != "cycleway"].index.tolist()), include_groups = False)
        
            masks = self.projects_summary.loc[:,'links_id_to_complete'].apply(
            lambda x: get_mask(x, self.routes_summary.loc[subsample_idxs,'links_id_to_complete'], alpha)
            )
        
            self.projects_summary.loc[:, 'n_near_completed_routes'] = masks.apply(sum)
            self.projects_summary.loc[:, 'near_completed_routes'] = masks.apply(
            lambda m: self.routes_summary.loc[subsample_idxs].loc[m].index
            )
            
        else:
            masks = self.routes_summary.loc[subsample_idxs,'links_id_to_complete'].apply(
            lambda x: get_mask(x, self.routes_summary.loc[subsample_idxs,'links_id_to_complete'], alpha)
            )
        
            self.routes_summary.loc[subsample_idxs, 'n_near_completed_routes'] = masks.apply(sum)
            self.routes_summary.loc[subsample_idxs, 'near_completed_routes'] = masks.apply(
            lambda m: self.routes_summary.loc[subsample_idxs].loc[m].index
            )
        
        
        self.routes_summary.loc[subsample_idxs,'fpkm'] = grouped_route_links.apply(
            lambda x: ((x.loc[x["highway"] != "cycleway", "flow"]*
                       x.loc[x["highway"] != "cycleway", "length"]).sum()/x.loc[x["highway"] != "cycleway", "length"].sum())
            if x.loc[x["highway"] != "cycleway", "length"].sum() != 0 else 0 ,include_groups=False)
        self.routes_summary.loc[subsample_idxs,'normalized_fpkm'] = self.routes_summary.loc[subsample_idxs,'fpkm']/self.routes_summary['fpkm'].sum()
    
        if not 'sl_dist' in self.routes_summary.columns:
            o_nodes_idxs = self.routes.loc[subsample_idxs,'nodes'].str[0]
            d_nodes_idxs = self.routes.loc[subsample_idxs,'nodes'].str[-1]
            self.routes_summary.loc[subsample_idxs,'sl_dist'] = self.n_pot_nodes.loc[o_nodes_idxs].distance(self.n_pot_nodes.loc[d_nodes_idxs],align = False).values
        self.routes_summary.loc[subsample_idxs,'tort'] = self.routes_summary.loc[subsample_idxs,
        'total_length']/self.routes_summary.loc[subsample_idxs,'sl_dist']
    
        if self.projects is not None:
            
            transit_min = grouped_projects['transit_min_dist'].min()
            self.projects_summary['transit_min_dist'] = self.projects_summary.index.map(transit_min)
        
            self.cycleway_nodes = set(self.n_pot_edges[self.n_pot_edges.highway == 'cycleway'].index.get_level_values('u')).union(
                set(self.n_pot_edges[self.n_pot_edges.highway == 'cycleway'].index.get_level_values('v')))
        
            
            projects_nodes = self.projects.groupby('project').apply(lambda p: set(p.index.get_level_values('u'))|set(p.index.get_level_values('v')),include_groups=False)
            
            self.projects_summary.loc[:, 'connects_existing_network'] = (
                projects_nodes.apply(lambda nodes: bool(set(nodes) & self.cycleway_nodes))
            )
            
            self.projects_summary.loc[:,'generalized_benefit']=self.projects_summary.loc[:,'n_near_completed_routes']/self.projects_summary.loc[:,'length_street']
            
            if weight is not None:
                self.projects_summary.loc[:,'generalized_benefit']*= self.projects_summary['near_completed_routes'].apply(lambda x: self.sample.loc[x,weight].sum())
            
            self.projects_summary.loc[:,'generalized_benefit']+=((self.projects_summary.loc[:]['transit_min_dist']<proximity_dist)*beta_transit*np.std(self.projects_summary.loc[:,'generalized_benefit'])
                                                                            +self.projects_summary.loc[:]['connects_existing_network']*beta_bikeway*np.std(self.projects_summary.loc[:,'generalized_benefit']))
        else:
    
            transit_min = grouped_route_links['transit_min_dist'].min()
            self.routes_summary['transit_min_dist'] = self.routes_summary.index.map(transit_min)
        
            self.cycleway_nodes = set(self.n_pot_edges[self.n_pot_edges.highway == 'cycleway'].index.get_level_values('u')).union(
                set(self.n_pot_edges[self.n_pot_edges.highway == 'cycleway'].index.get_level_values('v')))
            route_nodes = self.routes.loc[subsample_idxs, 'nodes']
            self.routes_summary.loc[subsample_idxs, 'connects_existing_network'] = (
                route_nodes.apply(lambda nodes: bool(set(nodes) & self.cycleway_nodes))
            )
            
            self.routes_summary.loc[subsample_idxs,'generalized_benefit']=self.routes_summary.loc[subsample_idxs]['n_near_completed_routes']/self.routes_summary.loc[subsample_idxs,'length_street']
            
            if weight is not None:
                self.routes_summary.loc[subsample_idxs,'generalized_benefit']*= self.routes_summary.loc[subsample_idxs,weight] 
            
            self.routes_summary.loc[subsample_idxs,'generalized_benefit']+=((self.routes_summary.loc[subsample_idxs]['transit_min_dist']<proximity_dist)*beta_transit*np.std(self.routes_summary.loc[subsample_idxs,'generalized_benefit'])
                                                                            +self.routes_summary.loc[subsample_idxs]['connects_existing_network']*beta_bikeway*np.std(self.routes_summary.loc[subsample_idxs,'generalized_benefit']))

    

    def plot_routes(self,network):

        if self.all_routes_edges is None:
            self.get_routes_edges(network)
        

        self.sample = self.sample.set_geometry('orig')
        self.map_routes = self.sample.explore(color='blue', name='orig')
        self.sample = self.sample.set_geometry('dest')
        map2 = self.sample.explore(color='red', name='dest', m=self.map_routes)
        map3 = self.boundaries.explore(m = self.map_routes, name = 'boundaries', fill = False)
        map4 = self.n_ex_edges.explore(color='black', name='existing', m=self.map_routes,style_kwds={'opacity': 0.3})
        map5 = self.n_pot_edges.explore(color='black', name='potential', m=self.map_routes,style_kwds={'opacity': 0.3})
        self.all_routes_edges['route_n'] = self.all_routes_edges['route_number'].astype(str)
        map8 = self.all_routes_edges.explore(column = 'route_n',cmap = 'gist_rainbow', name = 'routes', m = self.map_routes, legend = False
                                            ,style_kwds={'weight': 5})
        folium.LayerControl().add_to(self.map_routes)
        display(self.map_routes)


    def compute_routes_replace(self,network,turn_penalties,idxs,weight = 'gencost'):
        subsample = self.sample.loc[idxs]
        o_nodes,o_dists = ox.nearest_nodes(network,subsample.orig.x.values,subsample.orig.y.values, return_dist=True)
        d_nodes,d_dists = ox.nearest_nodes(network,subsample.dest.x.values,subsample.dest.y.values, return_dist=True)
        if not turn_penalties:
            routes = ox.shortest_path(network, o_nodes, d_nodes, weight=weight)
        elif turn_penalties:
            routes = []
            for source, target in zip(o_nodes, d_nodes):
                route = shortest_path_turn_penalty(network, source, target, weight, self.turn_penalties)
                routes.append(route)
        old_routes = self.routes.loc[idxs].copy()
        self.routes.loc[idxs,'nodes']=pd.Series(data = routes,index=idxs)
        changed_mask = (self.routes.loc[idxs].dropna(subset='nodes') == old_routes.dropna(subset='nodes'))
        print(f"{len(changed_mask[changed_mask.nodes == False])}  routes changed, replacing...")
        self.iterations_description.loc[self.iterations_description.index[-1], 'n_changed_routes'] = len(changed_mask[changed_mask.nodes == False])
        changed_idxs = changed_mask[changed_mask.nodes == False].index.values.tolist()
        route_edges = []
        if len(changed_idxs)==0:
            return None
        for idx in tqdm(changed_idxs):
                if self.routes.loc[idx].nodes is not None:
                    if len(self.routes.loc[idx].nodes)>1:
                        edges = ox.routing.route_to_gdf(network,self.routes.loc[idx].nodes)
                        edges['route_number'] = idx
                        route_edges.append(edges)
        newroutes = pd.concat(route_edges)
        self.all_routes_edges = self.all_routes_edges[~self.all_routes_edges['route_number'].isin(changed_idxs)]
        self.all_routes_edges = pd.concat([self.all_routes_edges,newroutes])
        
    def run_algo(self,n_iter = 1000, budget = 10000):
        self.reset_routes()
        self.compute_routes(self.n_pot)
        self.get_routes_edges(self.n_pot)
 
        for i in tqdm(range(n_iter)):
            clear_output(wait = True)
            self.compute_routes_summary()
            preconnected = self.routes_summary[(self.routes_summary['length_cycleway']>0)&(self.routes_summary['length_street']>0)]
            route_id_to_add = preconnected[preconnected['length_street'] == preconnected['length_street'].min()].index[0]
            print('adding ',route_id_to_add)
            route_edges = self.all_routes_edges[self.all_routes_edges.route_number == int(route_id_to_add)]
            route_edges_to_add = route_edges[route_edges.highway!='cycleway']
    
            edges_id_to_add = route_edges_to_add.index
            self.n_pot_edges.loc[edges_id_to_add,'highway']='cycleway'
            self.n_pot_edges.loc[edges_id_to_add,'build_iter'] = np.max(self.n_pot_edges.build_iter)+1
            self.n_ex_edges = self.n_pot_edges[self.n_pot_edges.highway == 'cycleway']
    
            self.weight_network()
            self.n_pot = ox.graph_from_gdfs(self.n_pot_nodes,self.n_pot_edges)
    
            recompute_polygon = route_edges.buffer(500).union_all()
            recompute_polygon = gpd.GeoDataFrame([recompute_polygon]).rename(columns={0:'geometry'}).set_geometry('geometry')
            recompute_polygon = recompute_polygon.set_crs(self.crs)
            
            o_int = recompute_polygon.sjoin(self.sample.set_geometry('orig')).index_right.tolist()
            d_int = recompute_polygon.sjoin(self.sample.set_geometry('dest')).index_right.tolist()
    
            recompute_idxs = list(set(o_int+d_int))

            m = recompute_polygon.explore()
            self.sample.loc[o_int].explore(m = m, color = 'yellow')
            display(m)
            break
    
            if len(recompute_idxs) == 0:
                continue
            else:
                print('recomputing ',len(recompute_idxs))
                self.compute_routes_replace(self.n_pot,recompute_idxs)
                
    def reset_network(self):
        self.n_ex = self.n_ex_og.copy()
        self.n_ex_nodes,self.n_ex_edges = ox.graph_to_gdfs(self.n_ex)
        self.n_pot = self.n_pot_og.copy()
        self.n_pot_nodes,self.n_pot_edges = ox.graph_to_gdfs(self.n_pot)
        self.iterations_description = self.iterations_description.iloc[0:0]
        
    def run_algo2(self, budget, savepath,beta_age, mu_age,scale_age,beta_transit,beta_bikeway,alpha,ltp,rtp,icp,tsp,build_dmin,weight = None,
              plot = False, buffer = 350, cost_reduction_factor=0.9, proximity_dist = 10, backup_every = 100,metric = 'completion',turn_penalties = False,recompute_routes = True):
        added_routes = []
        i=0
        
        if turn_penalties:
            missing = [name for name, value in {
                'ltp': ltp,
                'rtp': rtp,
                'icp': icp,
                'tsp':tsp
            }.items() if value is None]
    
            if missing:
                raise ValueError(
                    f"When `advanced=True`, the following arguments are required: {', '.join(missing)}"
                )
            
        if np.max(self.n_pot_edges.build_iter) == 0:
            os.makedirs(savepath,exist_ok=True)
            params = {
            "budget": budget,
            "beta_age": beta_age,
            "mu_age": mu_age,
            "scale_age": scale_age,
            "beta_transit": beta_transit,
            "beta_bikeway": beta_bikeway,
            "cycleway_reduction_factor": cost_reduction_factor,
            "proximity_dist": proximity_dist,
            "buffer": buffer,
            "rtp": ltp,
            "ltp": rtp,
            "icp": icp,
            "tsp":tsp,
            "alpha":alpha,
            "metric":metric}
            df_params = pd.DataFrame(list(params.items()), columns=["Parameter", "Value"])
            df_params.to_csv(savepath + "/params.csv", index=False)
            
            print('Initializing, computing desire boxes...')
            if 'bbox' not in self.sample.columns:
                self.od_bbox(buffer = buffer)
            self.n_pot_edges['gencost'] = self.n_pot_edges['length']
            self.weight_network(cycleway_reduc_factor=cost_reduction_factor, grades=True)
            # self.get_personal_spatial_features(mu_age,scale_age)
            if turn_penalties:
                self.get_turn_penalties(self.n_pot,icp = icp,ltp = ltp,rtp = rtp,tsp = tsp)
            self.reset_routes()
            self.evol = []
            print('Computing routes...')
            if not turn_penalties:
                self.compute_routes(self.n_pot)
            elif turn_penalties:
                self.compute_routes_turn_penalty(self.n_pot)
            self.get_routes_edges(self.n_pot)
            print('Computing route metrics...')
            self.compute_routes_summary(beta_age = beta_age,
                                        mu_age = mu_age,
                                        scale_age = scale_age,
                                        beta_transit = beta_transit,
                                        beta_bikeway = beta_bikeway,
                                        alpha = alpha,
                                        proximity_dist = proximity_dist,
                                        weight = weight)
            self.evol.append(self.routes_summary.copy())
        else:
            i=np.max(self.n_pot_edges.build_iter)
        while self.n_ex_edges[self.n_ex_edges.build_iter > 0].length.sum()/1000 < budget:
            
            if (i%backup_every==0):
                self.save_network(savepath = savepath,i=i,show = False)
                self.routes_summary.to_csv(savepath + f"/summary_{i}.csv", index=False)
                self.iterations_description.to_csv(savepath+ "/iterations_summary.csv")
            i+=1
            if plot:
                if i==0:
                    fig, ax = plt.subplots(figsize = (5,5))
                    ax.axis('off')
            clear_output(wait = True)
            print(f'Built {self.n_ex_edges[self.n_ex_edges.build_iter > 0].length.sum()/1000}/{str(budget)} km (iteration {i})')
    
            rank=0
            route_edges_to_add = []
            while len(route_edges_to_add)==0:
                if metric == 'random':
                    route_id_to_add = self.routes_summary[(self.routes_summary.length_street>0)&(self.routes_summary.length_cycleway>0)].sample().index[0]
                    print('Adding route ', route_id_to_add, f'(length = {self.routes_summary.loc[route_id_to_add,"length_street"]})')
        
                elif metric == 'fpkm':
                    route_id_to_add = self.routes_summary[(self.routes_summary.length_street>0)&(self.routes_summary.length_cycleway>0)].sort_values('normalized_fpkm',ascending = False).iloc[[rank]].index.values[0]
                    print('Adding route ', route_id_to_add, f'(normalized fpkm = {self.routes_summary.loc[route_id_to_add,"normalized_fpkm"]},length = {self.routes_summary.loc[route_id_to_add,"length_street"]})')
        
                elif metric == 'completion':
                    if self.projects is not None:
                        project_idx = self.projects_summary.sort_values('generalized_benefit',ascending = False).iloc[[rank]].index.values[0]
                    else:
                        route_id_to_add = self.routes_summary[(self.routes_summary.length_street>0)].sort_values('generalized_benefit',ascending = False).iloc[[rank]].index.values[0]
                
                elif metric == 'fill':
                    route_id_to_add = self.routes_summary[(self.routes_summary.length_street>0)&(self.routes_summary.prop_cycleway<0.1)].sort_values('generalized_benefit',ascending = False).iloc[[rank]].index.values[0]
                    print('Adding route ', route_id_to_add, f'(benefit = {self.routes_summary.loc[route_id_to_add,"generalized_benefit"]},length = {self.routes_summary.loc[route_id_to_add,"length_street"]})')
    
                if self.projects is not None:
                    edges_id_to_add = self.projects[self.projects['project'] == project_idx].index
                    potential_edges = split_components(self.n_pot_edges.loc[edges_id_to_add][
                                                   (self.n_pot_edges.loc[edges_id_to_add].highway!='cycleway')],self.cycleway_nodes)
                else:
                    potential_edges = split_components(self.all_routes_edges[(self.all_routes_edges.route_number == int(route_id_to_add))&
                                                       (self.all_routes_edges.highway!='cycleway')],self.cycleway_nodes)
                route_edges_to_add = potential_edges[(potential_edges.component_length>build_dmin)|(potential_edges.both_connected)]
                rank+=1
    
            if self.projects is not None:
                edges_id_to_add = self.projects[self.projects['project'] == project_idx].index
                self.iterations_description.loc[i,'generalized_benefit'] = self.projects_summary.loc[project_idx,"generalized_benefit"]
                self.iterations_description.loc[i,'n_near_completed_routes'] = self.projects_summary.loc[project_idx,"n_near_completed_routes"]
            else:
                edges_id_to_add = route_edges_to_add.index
                self.iterations_description.loc[i,'generalized_benefit'] = self.routes_summary.loc[route_id_to_add,"generalized_benefit"]
                self.iterations_description.loc[i,'n_near_completed_routes'] = self.routes_summary.loc[route_id_to_add,"n_near_completed_routes"]
            self.n_pot_edges.loc[edges_id_to_add,'highway']='cycleway'
            self.n_pot_edges.loc[edges_id_to_add,'build_iter'] = np.max(self.n_pot_edges.build_iter)+1
            self.n_ex_edges = self.n_pot_edges[self.n_pot_edges.highway == 'cycleway']
            
            
            added_length = self.n_pot_edges.loc[edges_id_to_add]['length'].sum()
            self.iterations_description.loc[i,'added_length'] = added_length
            
    
            if self.projects is not None:
                print('Adding', project_idx, f'(benefit = {self.projects_summary.loc[project_idx,"generalized_benefit"]},length = {added_length})')
            else:
                print('Adding route ', route_id_to_add, f'(benefit = {self.routes_summary.loc[route_id_to_add,"generalized_benefit"]},length = {added_length})')
    
            
            self.weight_network(cost_reduction_factor, grades = True)
            routes_int_idxs = self.associated_links.loc[edges_id_to_add].values
            routes_int_idxs = list(set(routes_int_idxs) & set(self.sample.index.values))
       
            
            
            if self.projects is None:
                self.all_routes_edges.loc[edges_id_to_add,'highway']='cycleway'
                added_routes.append(route_id_to_add)
                subsample_idxs = list(set(routes_int_idxs)|set([route_id_to_add]))
            else:
                used_edges_added = self.all_routes_edges.index.intersection(edges_id_to_add)
                self.all_routes_edges.loc[used_edges_added,'highway']='cycleway'
                subsample_idxs = list(set(routes_int_idxs))
            if recompute_routes:   
                if len(routes_int_idxs) == 0:
                    self.compute_routes_replace(self.n_pot,turn_penalties,[route_id_to_add])
                else:
                    print('recomputing ',len(routes_int_idxs)/len(self.sample),f'({len(routes_int_idxs)}/{len(self.sample)})')
                    self.compute_routes_replace(self.n_pot,turn_penalties,list(set(routes_int_idxs)-set(added_routes)))
                self.iterations_description.loc[i,'n_recomputed_routes'] = len(routes_int_idxs)
                
            print('Computing route metrics...')    
            self.compute_routes_summary(beta_age = beta_age,
                                        mu_age = mu_age,
                                        scale_age = scale_age,
                                        beta_transit = beta_transit,
                                        beta_bikeway = beta_bikeway,
                                        proximity_dist = proximity_dist,
                                        alpha = alpha,
                                        weight = weight,
                                        subsample_idxs = subsample_idxs)
            self.evol.append(self.routes_summary.copy())
        
        self.save_network(savepath = savepath,i='final')
        self.routes_summary.to_csv(savepath + f"/summary_{i}.csv", index=False)
        self.iterations_description.to_csv(savepath+ "/iterations_summary.csv")




    def add_edge_bearing(self):
        n_pot_unproj = ox.projection.project_graph(self.n_pot,to_crs=4326)
        n_pot_angle = ox.bearing.add_edge_bearings(n_pot_unproj)
        nodes,edges = ox.graph_to_gdfs(n_pot_angle)
        self.n_pot_edges['bearing'] = edges['bearing'].fillna(0)
        self.n_pot = ox.graph_from_gdfs(self.n_pot_nodes,self.n_pot_edges)
        self.n_pot_og = ox.graph_from_gdfs(self.n_pot_nodes.copy(),self.n_pot_edges.copy())
        
    def od_bbox(self,buffer):
        def compute_bounding_box(row):
            min_x = min(row["orig"].x, row["dest"].x)
            max_x = max(row["orig"].x, row["dest"].x)
            min_y = min(row["orig"].y, row["dest"].y)
            max_y = max(row["orig"].y, row["dest"].y)
            return box(min_x, min_y, max_x, max_y).buffer(buffer)
            
        
        def get_links_bbox(row):
            links = gpd.sjoin(row,self.n_pot_edges,how = 'left',
                              predicate = 'intersects').set_index(['u', 'v', 'key'])['ipere']
    
        n_pot_unproj = ox.projection.project_graph(self.n_pot,to_crs=4326)
        n_pot_angle = ox.bearing.add_edge_bearings(ox.convert.to_undirected(n_pot_unproj))
        nodes,edges = ox.graph_to_gdfs(n_pot_angle)
        edges = edges.dropna(subset=['bearing'])
        # lengths = pd.merge(edges,self.n_pot_edges,how = 'inner')['length']
        dom_angle = np.average(edges.bearing.values%90,weights = edges['length'])
        print('Dominating angle:', dom_angle)
        # dom_angle = 0
        centroid = self.sample.union_all().centroid
        rotated_sample_o = self.sample.orig.rotate(dom_angle,centroid).explode()
        rotated_sample_d = self.sample.dest.rotate(dom_angle,centroid).explode()
        rotated_sample_o.name = 'orig'
        rotated_sample_d.name = 'dest'
        rotated_sample = pd.concat([rotated_sample_o,rotated_sample_d],axis = 1)
        rotated_sample = gpd.GeoDataFrame(rotated_sample,geometry='orig')
        rotated_sample["bbox"] = rotated_sample.apply(compute_bounding_box, axis=1)
        rotated_sample.set_crs(self.crs)
        self.sample['bbox'] = rotated_sample.bbox.rotate(-dom_angle,centroid)
        self.sample = self.sample.set_geometry('bbox').set_crs(self.crs)
        self.associated_links = gpd.sjoin(self.sample,self.n_pot_edges,how = 'left',
                              predicate = 'intersects').set_index(['u', 'v', 'key'])['ipere']
        
    def weight_network(self,cycleway_reduc_factor=0.9, grades = False):
        self.n_pot_edges['gencost'] = self.n_pot_edges['length']*self.n_pot_edges["highway"].apply(
            lambda x: cycleway_reduc_factor if x == "cycleway" else 1)
        if grades:
            self.n_pot_edges.loc[self.n_pot_edges.length>15,'gencost'] *= (1*(self.n_pot_edges.grade<0.02)+1.371*((self.n_pot_edges.grade>=0.02)&(self.n_pot_edges.grade<0.04))+
                                      2.203*((self.n_pot_edges.grade>=0.04)&(self.n_pot_edges.grade<0.06))+
                                      4.239*((self.n_pot_edges.grade>=0.06)))
                                     
        self.n_pot = ox.graph_from_gdfs(self.n_pot_nodes,self.n_pot_edges)
    
    def add_edge_grades(self,elevation_filepath):
        self.n_pot = ox.elevation.add_node_elevations_raster(self.n_pot,elevation_filepath)
        self.n_pot = ox.elevation.add_edge_grades(self.n_pot,add_absolute=True)
        self.n_ex = ox.elevation.add_node_elevations_raster(self.n_ex,elevation_filepath)
        self.n_ex = ox.elevation.add_edge_grades(self.n_ex,add_absolute=True)
        self.n_pot_nodes,self.n_pot_edges = ox.graph_to_gdfs(self.n_pot)
        self.n_ex_nodes,self.n_ex_edges = ox.graph_to_gdfs(self.n_ex)
        self.n_pot_edges['build_iter'] = 0
        self.n_ex_edges['build_iter'] = 0
        self.n_pot_og = ox.graph_from_gdfs(self.n_pot_nodes.copy(),self.n_pot_edges.copy())
        self.n_ex_og = ox.graph_from_gdfs(self.n_ex_nodes.copy(),self.n_ex_edges.copy())
    
    def display_network(self, tiles = 'CartoDB Positron',savepath = None, max_iter = None,color = 'red',show = True):
        built_edges = self.n_ex_edges[self.n_ex_edges.build_iter > 0]
        existing_edges = self.n_ex_edges[self.n_ex_edges.build_iter == 0]
        if max_iter is not None:
            built_edges = built_edges[built_edges.build_iter<=max_iter]
              
        new_network = built_edges.explore(tiles = tiles,
            column='build_iter',
            cmap='viridis_r',
            style_kwds={'weight': 4},
            legend=True
        )
        
        existing_edges.explore(
            color=color,
            style_kwds={'weight': 2, 'opacity': 0.5},
            m=new_network
        )
        
        dummy = existing_edges.head(1).copy()
        dummy['Legend'] = 'Existing network'
        dummy.explore(
            column='Legend',
            cmap=[color],
            m=new_network,
            legend=True
        )
    
        Fullscreen(position='topright').add_to(new_network)
        smooth_zoom_js = """
        <script>
            var map = {{this._parent.get_name()}};
            map.options.zoomSnap = 0.1;
            map.options.zoomDelta = 0.1;
        </script>
        """
        
        new_network.get_root().html.add_child(folium.Element(smooth_zoom_js))
        if show:
            display(new_network)
            print(f'length = {built_edges.length.sum()/1000} km')
        if savepath is not None:
            new_network.save(savepath)
    
        return new_network
    
        
    def compute_routes(self, network, weight = 'gencost'):
        network = ox.projection.project_graph(network, to_crs = self.crs)
        o_nodes,o_dists = ox.nearest_nodes(network,self.sample.orig.x.values,self.sample.orig.y.values, return_dist=True)
        d_nodes,d_dists = ox.nearest_nodes(network,self.sample.dest.x.values,self.sample.dest.y.values, return_dist=True)
        routes = ox.shortest_path(network, o_nodes, d_nodes, weight=weight,cpus = None)
        self.routes = pd.DataFrame(data = {'nodes': routes}, index = self.sample.index)
        
    def compute_routes_turn_penalty(self, network, weight = 'gencost'):
        network = ox.projection.project_graph(network, to_crs = self.crs)
        o_nodes,o_dists = ox.nearest_nodes(network,self.sample.orig.x.values,self.sample.orig.y.values, return_dist=True)
        d_nodes,d_dists = ox.nearest_nodes(network,self.sample.dest.x.values,self.sample.dest.y.values, return_dist=True)
        routes = []
        for source, target in zip(o_nodes, d_nodes):
            route = shortest_path_turn_penalty(network, source, target, weight, self.turn_penalties)
            routes.append(route)
        self.routes = pd.DataFrame(data = {'nodes': routes}, index = self.sample.index)
        
    def save_network(self,savepath,i,show = True):
        print('Network backup at '+ savepath)
        ox.io.save_graphml(self.n_pot,filepath = savepath+f'/network_it_{i}.graphml')
        self.display_network(savepath = savepath+f'/network_it_{i}.html',show = show)
    
    def get_personal_spatial_features(self,mu_age,scale_age):
        self.n_pot_edges['bikeway_min_dist'] = self.n_pot_edges.shortest_line(self.n_ex_edges.union_all()).length
        self.n_pot_edges['transit_min_dist'] = self.n_pot_edges.shortest_line(self.transit_stops.union_all()).length
        self.n_pot = ox.graph_from_gdfs(self.n_pot_nodes,self.n_pot_edges)
        age_multiplier = scipy.stats.gumbel_l(loc = mu_age,scale = scale_age).pdf(self.sample['age'])
        age_multiplier/=np.max(age_multiplier)
        age_multiplier = 2-age_multiplier
        self.sample['age_multiplier'] = age_multiplier
    
    def get_turn_penalties(self,network,icp,ltp,rtp,tsp):
        self.turn_penalties = penalty_turns(network,left_turn_penalty=ltp,right_turn_penalty=rtp,intersection_crossing_penalty=icp,traffic_signals_penalty=tsp)

    def cross_gdf_buildable(self,gdf):
        gdf['buffer'] = gdf.buffer(5)
        self.n_pot_edges['buffer'] = self.n_pot_edges.buffer(5)
        gdf['buffer_gdf'] = gdf.buffer(5)
        self.n_pot_edges['buffer_osm'] = self.n_pot_edges.buffer(5)
        
        overlay = gpd.overlay(gdf.set_geometry('buffer').to_crs(self.crs),self.n_pot_edges.reset_index().set_geometry('buffer'),how = 'intersection')
        overlay['overlay_ratio'] = overlay.geometry.area/overlay.buffer_osm.area
        overlay = overlay[overlay.overlay_ratio>0.5]
        edges_overlay = self.n_pot_edges.loc[overlay.loc[overlay.groupby(['u','v','key'])['overlay_ratio'].idxmax()].set_index(['u','v','key']).index]
        self.n_pot_edges.loc[edges_overlay.index,'buildable']=True
        self.n_pot_edges.loc[:,'buildable'] = self.n_pot_edges.loc[:,'buildable'].fillna(False)
        self.n_pot = ox.graph_from_gdfs(self.n_pot_nodes,self.n_pot_edges)
        self.projects = group_street_components(self.n_pot_edges[self.n_pot_edges.buildable == True],name_col = 'name')
        self.n_pot_og = ox.graph_from_gdfs(self.n_pot_nodes.copy(),self.n_pot_edges.copy())
        self.n_ex_og = ox.graph_from_gdfs(self.n_ex_nodes.copy(),self.n_ex_edges.copy())
    