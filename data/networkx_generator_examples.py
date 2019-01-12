import networkx as nx

def draw( graph,
          output_folder='/app/host',
          output_name='graph',
          file_format='png' ):
    """ Draw the markov network.  Optionally, pass an order to
        convert the graph into a directed graph.  This can make
        the graph more visually appealing.

    Args:
        node_ordering - If None, will be however the nodes
        are ordered under the hood

    Returns:
        render - The graphviz render
    """
    graph = nx.nx_agraph.to_agraph( graph )

    output_folder = output_folder if output_folder[-1] != '/' else output_folder[:-1]
    output_name = output_name if '/' not in output_name else output_name.replace( '/', '' )
    file_format = file_format.replace( '.', '' )

    graph.draw( '%s/%s.%s'%( output_folder, output_name, file_format ), prog='neato' )

def draw_graph_atlas():
    i = 2
    draw_graph_atlas = nx.generators.graph_atlas( i )
    draw( draw_graph_atlas, output_folder='/app/host/networkx_generators', output_name='draw_graph_atlas' )

def draw_graph_atlas_g():
    return # This is expensive apparently
    graph_atlas_g = nx.generators.graph_atlas_g()
    draw( graph_atlas_g, output_folder='/app/host/networkx_generators', output_name='graph_atlas_g' )

def draw_balanced_tree():
    r = 3
    h = 3
    balanced_tree = nx.generators.balanced_tree( r, h )
    draw( balanced_tree, output_folder='/app/host/networkx_generators', output_name='balanced_tree' )

def draw_barbell_graph():
    m1 = 3
    m2 = 5
    barbell_graph = nx.generators.barbell_graph( m1, m2 )
    draw( barbell_graph, output_folder='/app/host/networkx_generators', output_name='barbell_graph' )

def draw_complete_graph():
    n = 5
    complete_graph = nx.generators.complete_graph( n )
    draw( complete_graph, output_folder='/app/host/networkx_generators', output_name='complete_graph' )

def draw_complete_multipartite_graph():
    subset_sizes = [ 1, 2, 3, 4 ]
    complete_multipartite_graph = nx.generators.complete_multipartite_graph( *subset_sizes )
    draw( complete_multipartite_graph, output_folder='/app/host/networkx_generators', output_name='complete_multipartite_graph' )

def draw_circular_ladder_graph():
    n = 5
    circular_ladder_graph = nx.generators.circular_ladder_graph( n )
    draw( circular_ladder_graph, output_folder='/app/host/networkx_generators', output_name='circular_ladder_graph' )

def draw_circulant_graph():
    n = 5
    offsets = [ 1, 2 ]
    circulant_graph = nx.generators.circulant_graph( n, offsets )
    draw( circulant_graph, output_folder='/app/host/networkx_generators', output_name='circulant_graph' )

def draw_cycle_graph():
    n = 10
    cycle_graph = nx.generators.cycle_graph( n )
    draw( cycle_graph, output_folder='/app/host/networkx_generators', output_name='cycle_graph' )

def draw_dorogovtsev_goltsev_mendes_graph():
    n = 5
    dorogovtsev_goltsev_mendes_graph = nx.generators.dorogovtsev_goltsev_mendes_graph( n )
    draw( dorogovtsev_goltsev_mendes_graph, output_folder='/app/host/networkx_generators', output_name='dorogovtsev_goltsev_mendes_graph' )

def draw_empty_graph():
    n = 3
    empty_graph = nx.generators.empty_graph( n )
    draw( empty_graph, output_folder='/app/host/networkx_generators', output_name='empty_graph' )

def draw_full_rary_tree():
    r = 3
    n = 10
    full_rary_tree = nx.generators.full_rary_tree( r, n )
    draw( full_rary_tree, output_folder='/app/host/networkx_generators', output_name='full_rary_tree' )

def draw_ladder_graph():
    n = 5
    ladder_graph = nx.generators.ladder_graph( n )
    draw( ladder_graph, output_folder='/app/host/networkx_generators', output_name='ladder_graph' )

def draw_lollipop_graph():
    m = 4
    n = 5
    lollipop_graph = nx.generators.lollipop_graph( m, n )
    draw( lollipop_graph, output_folder='/app/host/networkx_generators', output_name='lollipop_graph' )

def draw_null_graph():
    null_graph = nx.generators.null_graph()
    draw( null_graph, output_folder='/app/host/networkx_generators', output_name='null_graph' )

def draw_path_graph():
    n = 10
    path_graph = nx.generators.path_graph( n )
    draw( path_graph, output_folder='/app/host/networkx_generators', output_name='path_graph' )

def draw_star_graph():
    n = 8
    star_graph = nx.generators.star_graph( n )
    draw( star_graph, output_folder='/app/host/networkx_generators', output_name='star_graph' )

def draw_trivial_graph():
    trivial_graph = nx.generators.trivial_graph()
    draw( trivial_graph, output_folder='/app/host/networkx_generators', output_name='trivial_graph' )

def draw_turan_graph():
    n = 6
    r = 3
    turan_graph = nx.generators.turan_graph( n, r )
    draw( turan_graph, output_folder='/app/host/networkx_generators', output_name='turan_graph' )

def draw_wheel_graph():
    n = 5
    wheel_graph = nx.generators.wheel_graph( n )
    draw( wheel_graph, output_folder='/app/host/networkx_generators', output_name='wheel_graph' )

def draw_margulis_gabber_galil_graph():
    n = 5
    margulis_gabber_galil_graph = nx.generators.margulis_gabber_galil_graph( n )
    draw( margulis_gabber_galil_graph, output_folder='/app/host/networkx_generators', output_name='margulis_gabber_galil_graph' )

def draw_chordal_cycle_graph():
    p = 5
    chordal_cycle_graph = nx.generators.chordal_cycle_graph( p )
    draw( chordal_cycle_graph, output_folder='/app/host/networkx_generators', output_name='chordal_cycle_graph' )

def draw_grid_2d_graph():
    m = 4
    n = 5
    grid_2d_graph = nx.generators.grid_2d_graph( m, n )
    draw( grid_2d_graph, output_folder='/app/host/networkx_generators', output_name='grid_2d_graph' )

def draw_grid_graph():
    dim = [ 2, 3, 4 ]
    grid_graph = nx.generators.grid_graph( dim )
    draw( grid_graph, output_folder='/app/host/networkx_generators', output_name='grid_graph' )

def draw_hexagonal_lattice_graph():
    m = 5
    n = 4
    hexagonal_lattice_graph = nx.generators.hexagonal_lattice_graph( m, n )
    draw( hexagonal_lattice_graph, output_folder='/app/host/networkx_generators', output_name='hexagonal_lattice_graph' )

def draw_hypercube_graph():
    n = 4
    hypercube_graph = nx.generators.hypercube_graph( n )
    draw( hypercube_graph, output_folder='/app/host/networkx_generators', output_name='hypercube_graph' )

def draw_triangular_lattice_graph():
    m = 5
    n = 4
    triangular_lattice_graph = nx.generators.triangular_lattice_graph( m, n )
    draw( triangular_lattice_graph, output_folder='/app/host/networkx_generators', output_name='triangular_lattice_graph' )

def draw_make_small_graph():
    return # ?
    make_small_graph = nx.generators.make_small_graph( graph_description )
    draw( make_small_graph, output_folder='/app/host/networkx_generators', output_name='make_small_graph' )

def draw_LCF_graph():
    n = 14
    shift_list = [ 5, -5 ]
    repeats = 7
    LCF_graph = nx.generators.LCF_graph( n, shift_list, repeats )
    draw( LCF_graph, output_folder='/app/host/networkx_generators', output_name='LCF_graph' )

def draw_bull_graph():
    bull_graph = nx.generators.bull_graph()
    draw( bull_graph, output_folder='/app/host/networkx_generators', output_name='bull_graph' )

def draw_chvatal_graph():
    chvatal_graph = nx.generators.chvatal_graph()
    draw( chvatal_graph, output_folder='/app/host/networkx_generators', output_name='chvatal_graph' )

def draw_cubical_graph():
    cubical_graph = nx.generators.cubical_graph()
    draw( cubical_graph, output_folder='/app/host/networkx_generators', output_name='cubical_graph' )

def draw_desargues_graph():
    desargues_graph = nx.generators.desargues_graph()
    draw( desargues_graph, output_folder='/app/host/networkx_generators', output_name='desargues_graph' )

def draw_diamond_graph():
    diamond_graph = nx.generators.diamond_graph()
    draw( diamond_graph, output_folder='/app/host/networkx_generators', output_name='diamond_graph' )

def draw_dodecahedral_graph():
    dodecahedral_graph = nx.generators.dodecahedral_graph()
    draw( dodecahedral_graph, output_folder='/app/host/networkx_generators', output_name='dodecahedral_graph' )

def draw_frucht_graph():
    frucht_graph = nx.generators.frucht_graph()
    draw( frucht_graph, output_folder='/app/host/networkx_generators', output_name='frucht_graph' )

def draw_heawood_graph():
    heawood_graph = nx.generators.heawood_graph()
    draw( heawood_graph, output_folder='/app/host/networkx_generators', output_name='heawood_graph' )

def draw_hoffman_singleton_graph():
    hoffman_singleton_graph = nx.generators.hoffman_singleton_graph()
    draw( hoffman_singleton_graph, output_folder='/app/host/networkx_generators', output_name='hoffman_singleton_graph' )

def draw_house_graph():
    house_graph = nx.generators.house_graph()
    draw( house_graph, output_folder='/app/host/networkx_generators', output_name='house_graph' )

def draw_house_x_graph():
    house_x_graph = nx.generators.house_x_graph()
    draw( house_x_graph, output_folder='/app/host/networkx_generators', output_name='house_x_graph' )

def draw_icosahedral_graph():
    icosahedral_graph = nx.generators.icosahedral_graph()
    draw( icosahedral_graph, output_folder='/app/host/networkx_generators', output_name='icosahedral_graph' )

def draw_krackhardt_kite_graph():
    krackhardt_kite_graph = nx.generators.krackhardt_kite_graph()
    draw( krackhardt_kite_graph, output_folder='/app/host/networkx_generators', output_name='krackhardt_kite_graph' )

def draw_moebius_kantor_graph():
    moebius_kantor_graph = nx.generators.moebius_kantor_graph()
    draw( moebius_kantor_graph, output_folder='/app/host/networkx_generators', output_name='moebius_kantor_graph' )

def draw_octahedral_graph():
    octahedral_graph = nx.generators.octahedral_graph()
    draw( octahedral_graph, output_folder='/app/host/networkx_generators', output_name='octahedral_graph' )

def draw_pappus_graph():
    pappus_graph = nx.generators.pappus_graph()
    draw( pappus_graph, output_folder='/app/host/networkx_generators', output_name='pappus_graph' )

def draw_petersen_graph():
    petersen_graph = nx.generators.petersen_graph()
    draw( petersen_graph, output_folder='/app/host/networkx_generators', output_name='petersen_graph' )

def draw_sedgewick_maze_graph():
    sedgewick_maze_graph = nx.generators.sedgewick_maze_graph()
    draw( sedgewick_maze_graph, output_folder='/app/host/networkx_generators', output_name='sedgewick_maze_graph' )

def draw_tetrahedral_graph():
    tetrahedral_graph = nx.generators.tetrahedral_graph()
    draw( tetrahedral_graph, output_folder='/app/host/networkx_generators', output_name='tetrahedral_graph' )

def draw_truncated_cube_graph():
    truncated_cube_graph = nx.generators.truncated_cube_graph()
    draw( truncated_cube_graph, output_folder='/app/host/networkx_generators', output_name='truncated_cube_graph' )

def draw_truncated_tetrahedron_graph():
    truncated_tetrahedron_graph = nx.generators.truncated_tetrahedron_graph()
    draw( truncated_tetrahedron_graph, output_folder='/app/host/networkx_generators', output_name='truncated_tetrahedron_graph' )

def draw_tutte_graph():
    tutte_graph = nx.generators.tutte_graph()
    draw( tutte_graph, output_folder='/app/host/networkx_generators', output_name='tutte_graph' )

def draw_fast_gnp_random_graph():
    n = 5
    p = 0.5
    fast_gnp_random_graph = nx.generators.fast_gnp_random_graph( n, p )
    draw( fast_gnp_random_graph, output_folder='/app/host/networkx_generators', output_name='fast_gnp_random_graph' )

def draw_gnp_random_graph():
    n = 5
    p = 0.5
    gnp_random_graph = nx.generators.gnp_random_graph( n, p )
    draw( gnp_random_graph, output_folder='/app/host/networkx_generators', output_name='gnp_random_graph' )

def draw_dense_gnm_random_graph():
    n = 5
    m = 3
    dense_gnm_random_graph = nx.generators.dense_gnm_random_graph( n, m )
    draw( dense_gnm_random_graph, output_folder='/app/host/networkx_generators', output_name='dense_gnm_random_graph' )

def draw_gnm_random_graph():
    n = 5
    m = 3
    gnm_random_graph = nx.generators.gnm_random_graph( n, m )
    draw( gnm_random_graph, output_folder='/app/host/networkx_generators', output_name='gnm_random_graph' )

def draw_erdos_renyi_graph():
    n = 5
    p = 0.5
    erdos_renyi_graph = nx.generators.erdos_renyi_graph( n, p )
    draw( erdos_renyi_graph, output_folder='/app/host/networkx_generators', output_name='erdos_renyi_graph' )

def draw_binomial_graph():
    n = 5
    p = 0.5
    binomial_graph = nx.generators.binomial_graph( n, p )
    draw( binomial_graph, output_folder='/app/host/networkx_generators', output_name='binomial_graph' )

def draw_newman_watts_strogatz_graph():
    n = 5
    k = 3
    p = 0.5
    newman_watts_strogatz_graph = nx.generators.newman_watts_strogatz_graph( n, k, p )
    draw( newman_watts_strogatz_graph, output_folder='/app/host/networkx_generators', output_name='newman_watts_strogatz_graph' )

def draw_watts_strogatz_graph():
    n = 5
    k = 3
    p = 0.5
    watts_strogatz_graph = nx.generators.watts_strogatz_graph( n, k, p )
    draw( watts_strogatz_graph, output_folder='/app/host/networkx_generators', output_name='watts_strogatz_graph' )

def draw_connected_watts_strogatz_graph():
    n = 5
    k = 3
    p = 0.5
    connected_watts_strogatz_graph = nx.generators.connected_watts_strogatz_graph( n, k, p )
    draw( connected_watts_strogatz_graph, output_folder='/app/host/networkx_generators', output_name='connected_watts_strogatz_graph' )

def draw_random_regular_graph():
    d = 4
    n = 5
    random_regular_graph = nx.generators.random_regular_graph( d, n )
    draw( random_regular_graph, output_folder='/app/host/networkx_generators', output_name='random_regular_graph' )

def draw_barabasi_albert_graph():
    n = 5
    m = 3
    barabasi_albert_graph = nx.generators.barabasi_albert_graph( n, m )
    draw( barabasi_albert_graph, output_folder='/app/host/networkx_generators', output_name='barabasi_albert_graph' )

def draw_extended_barabasi_albert_graph():
    n = 10
    m = 5
    p = 0.5
    q = 0.3
    extended_barabasi_albert_graph = nx.generators.extended_barabasi_albert_graph( n, m, p, q )
    draw( extended_barabasi_albert_graph, output_folder='/app/host/networkx_generators', output_name='extended_barabasi_albert_graph' )

def draw_powerlaw_cluster_graph():
    n = 10
    m = 5
    p = 0.5
    powerlaw_cluster_graph = nx.generators.powerlaw_cluster_graph( n, m, p )
    draw( powerlaw_cluster_graph, output_folder='/app/host/networkx_generators', output_name='powerlaw_cluster_graph' )

def draw_random_kernel_graph():
    n = 10
    kernel_integral = lambda u, w, z : u * ( z - w )
    random_kernel_graph = nx.generators.random_kernel_graph( n, kernel_integral )
    draw( random_kernel_graph, output_folder='/app/host/networkx_generators', output_name='random_kernel_graph' )

def draw_random_lobster():
    n = 10
    p1 = 0.5
    p2 = 0.5
    random_lobster = nx.generators.random_lobster( n, p1, p2 )
    draw( random_lobster, output_folder='/app/host/networkx_generators', output_name='random_lobster' )

def draw_random_shell_graph():
    constructor = [ ( 10, 20, 0.8 ), ( 20, 40, 0.8 ) ]
    random_shell_graph = nx.generators.random_shell_graph( constructor )
    draw( random_shell_graph, output_folder='/app/host/networkx_generators', output_name='random_shell_graph' )

def draw_random_powerlaw_tree():
    n = 5
    random_powerlaw_tree = nx.generators.random_powerlaw_tree( n )
    draw( random_powerlaw_tree, output_folder='/app/host/networkx_generators', output_name='random_powerlaw_tree' )

def draw_random_powerlaw_tree_sequence():
    return # ?
    n = 5
    random_powerlaw_tree_sequence = nx.generators.random_powerlaw_tree_sequence( n )
    draw( random_powerlaw_tree_sequence, output_folder='/app/host/networkx_generators', output_name='random_powerlaw_tree_sequence' )

def draw_duplication_divergence_graph():
    n = 10
    p = 0.5
    duplication_divergence_graph = nx.generators.duplication_divergence_graph( n, p )
    draw( duplication_divergence_graph, output_folder='/app/host/networkx_generators', output_name='duplication_divergence_graph' )

def draw_partial_duplication_graph():
    N = 10
    n = 5
    p = 0.5
    q = 0.5
    partial_duplication_graph = nx.generators.partial_duplication_graph( N, n, p, q )
    draw( partial_duplication_graph, output_folder='/app/host/networkx_generators', output_name='partial_duplication_graph' )

def draw_configuration_model():
    return # ?
    configuration_model = nx.generators.configuration_model( deg_sequence )
    draw( configuration_model, output_folder='/app/host/networkx_generators', output_name='configuration_model' )

def draw_directed_configuration_model():
    return # ?
    directed_configuration_model = nx.generators.directed_configuration_model()
    draw( directed_configuration_model, output_folder='/app/host/networkx_generators', output_name='directed_configuration_model' )

def draw_expected_degree_graph():
    w = [10 for i in range(100)]
    expected_degree_graph = nx.generators.expected_degree_graph( w )
    draw( expected_degree_graph, output_folder='/app/host/networkx_generators', output_name='expected_degree_graph' )

def draw_havel_hakimi_graph():
    return
    deg_sequence = [ 2, 2, 4 ]
    havel_hakimi_graph = nx.generators.havel_hakimi_graph( deg_sequence )
    draw( havel_hakimi_graph, output_folder='/app/host/networkx_generators', output_name='havel_hakimi_graph' )

def draw_directed_havel_hakimi_graph():
    return
    in_deg_sequence = [ 2, 2, 4 ]
    return
    out_deg_sequence = [ 2, 2, 4 ]
    directed_havel_hakimi_graph = nx.generators.directed_havel_hakimi_graph( in_deg_sequence, out_deg_sequence )
    draw( directed_havel_hakimi_graph, output_folder='/app/host/networkx_generators', output_name='directed_havel_hakimi_graph' )

def draw_degree_sequence_tree():
    return
    deg_sequence = [ 2, 2, 4 ]
    degree_sequence_tree = nx.generators.degree_sequence_tree( deg_sequence )
    draw( degree_sequence_tree, output_folder='/app/host/networkx_generators', output_name='degree_sequence_tree' )

def draw_random_degree_sequence_graph():
    return
    sequence = [ 2, 2, 4 ]
    random_degree_sequence_graph = nx.generators.random_degree_sequence_graph( sequence )
    draw( random_degree_sequence_graph, output_folder='/app/host/networkx_generators', output_name='random_degree_sequence_graph' )

def draw_random_clustered_graph():
    joint_degree_sequence = [(1, 0), (1, 0), (1, 0), (2, 0), (1, 0), (2, 1), (0, 1), (0, 1)]
    random_clustered_graph = nx.generators.random_clustered_graph( joint_degree_sequence )
    draw( random_clustered_graph, output_folder='/app/host/networkx_generators', output_name='random_clustered_graph' )

def draw_gn_graph():
    n = 5
    gn_graph = nx.generators.gn_graph( n )
    draw( gn_graph, output_folder='/app/host/networkx_generators', output_name='gn_graph' )

def draw_gnr_graph():
    n = 10
    p = 0.5
    gnr_graph = nx.generators.gnr_graph( n, p )
    draw( gnr_graph, output_folder='/app/host/networkx_generators', output_name='gnr_graph' )

def draw_gnc_graph():
    n = 5
    gnc_graph = nx.generators.gnc_graph( n )
    draw( gnc_graph, output_folder='/app/host/networkx_generators', output_name='gnc_graph' )

def draw_random_k_out_graph():
    n = 8
    k = 3
    alpha = 0.5
    random_k_out_graph = nx.generators.random_k_out_graph( n, k, alpha )
    draw( random_k_out_graph, output_folder='/app/host/networkx_generators', output_name='random_k_out_graph' )

def draw_scale_free_graph():
    n = 5
    scale_free_graph = nx.generators.scale_free_graph( n )
    draw( scale_free_graph, output_folder='/app/host/networkx_generators', output_name='scale_free_graph' )

def draw_random_geometric_graph():
    n = 5
    radius = 3
    random_geometric_graph = nx.generators.random_geometric_graph( n, radius )
    draw( random_geometric_graph, output_folder='/app/host/networkx_generators', output_name='random_geometric_graph' )

def draw_soft_random_geometric_graph():
    n = 5
    radius = 3
    soft_random_geometric_graph = nx.generators.soft_random_geometric_graph( n, radius )
    draw( soft_random_geometric_graph, output_folder='/app/host/networkx_generators', output_name='soft_random_geometric_graph' )

def draw_geographical_threshold_graph():
    n = 5
    theta = 2
    geographical_threshold_graph = nx.generators.geographical_threshold_graph( n, theta )
    draw( geographical_threshold_graph, output_folder='/app/host/networkx_generators', output_name='geographical_threshold_graph' )

def draw_waxman_graph():
    n = 5
    waxman_graph = nx.generators.waxman_graph( n )
    draw( waxman_graph, output_folder='/app/host/networkx_generators', output_name='waxman_graph' )

def draw_navigable_small_world_graph():
    n = 5
    navigable_small_world_graph = nx.generators.navigable_small_world_graph( n )
    draw( navigable_small_world_graph, output_folder='/app/host/networkx_generators', output_name='navigable_small_world_graph' )

def draw_thresholded_random_geometric_graph():
    n = 6
    radius = 3
    theta = 1
    thresholded_random_geometric_graph = nx.generators.thresholded_random_geometric_graph( n, radius, theta )
    draw( thresholded_random_geometric_graph, output_folder='/app/host/networkx_generators', output_name='thresholded_random_geometric_graph' )

def draw_line_graph():
    return # ?
    line_graph = nx.generators.line_graph( G )
    draw( line_graph, output_folder='/app/host/networkx_generators', output_name='line_graph' )

def draw_inverse_line_graph():
    return # ?
    inverse_line_graph = nx.generators.inverse_line_graph( G )
    draw( inverse_line_graph, output_folder='/app/host/networkx_generators', output_name='inverse_line_graph' )

def draw_ego_graph():
    return
    n = 5
    ego_graph = nx.generators.ego_graph( G, n )
    draw( ego_graph, output_folder='/app/host/networkx_generators', output_name='ego_graph' )

def draw_stochastic_graph():
    return # ?
    stochastic_graph = nx.generators.stochastic_graph( G )
    draw( stochastic_graph, output_folder='/app/host/networkx_generators', output_name='stochastic_graph' )

def draw_uniform_random_intersection_graph():
    n = 6
    m = 4
    p = 0.5
    uniform_random_intersection_graph = nx.generators.uniform_random_intersection_graph( n, m, p )
    draw( uniform_random_intersection_graph, output_folder='/app/host/networkx_generators', output_name='uniform_random_intersection_graph' )

def draw_k_random_intersection_graph():
    n = 6
    m = 4
    k = 3
    k_random_intersection_graph = nx.generators.k_random_intersection_graph( n, m, k )
    draw( k_random_intersection_graph, output_folder='/app/host/networkx_generators', output_name='k_random_intersection_graph' )

def draw_general_random_intersection_graph():
    n = 6
    m = 4
    p = [ 0.5 for _ in range( m ) ]
    general_random_intersection_graph = nx.generators.general_random_intersection_graph( n, m, p )
    draw( general_random_intersection_graph, output_folder='/app/host/networkx_generators', output_name='general_random_intersection_graph' )

def draw_karate_club_graph():
    karate_club_graph = nx.generators.karate_club_graph()
    draw( karate_club_graph, output_folder='/app/host/networkx_generators', output_name='karate_club_graph' )

def draw_davis_southern_women_graph():
    davis_southern_women_graph = nx.generators.davis_southern_women_graph()
    draw( davis_southern_women_graph, output_folder='/app/host/networkx_generators', output_name='davis_southern_women_graph' )

def draw_florentine_families_graph():
    florentine_families_graph = nx.generators.florentine_families_graph()
    draw( florentine_families_graph, output_folder='/app/host/networkx_generators', output_name='florentine_families_graph' )

def draw_caveman_graph():
    l = 10
    k = 3
    caveman_graph = nx.generators.caveman_graph( l, k )
    draw( caveman_graph, output_folder='/app/host/networkx_generators', output_name='caveman_graph' )

def draw_connected_caveman_graph():
    l = 10
    k = 3
    connected_caveman_graph = nx.generators.connected_caveman_graph( l, k )
    draw( connected_caveman_graph, output_folder='/app/host/networkx_generators', output_name='connected_caveman_graph' )

def draw_relaxed_caveman_graph():
    l = 10
    k = 3
    p = 0.5
    relaxed_caveman_graph = nx.generators.relaxed_caveman_graph( l, k, p )
    draw( relaxed_caveman_graph, output_folder='/app/host/networkx_generators', output_name='relaxed_caveman_graph' )

def draw_random_partition_graph():
    sizes = [ 10, 10, 10 ]
    p_in = 0.25
    p_out = 0.01
    random_partition_graph = nx.generators.random_partition_graph( sizes, p_in, p_out )
    draw( random_partition_graph, output_folder='/app/host/networkx_generators', output_name='random_partition_graph' )

def draw_planted_partition_graph():
    l = 10
    k = 3
    p_in = 0.25
    p_out = 0.01
    planted_partition_graph = nx.generators.planted_partition_graph( l, k, p_in, p_out )
    draw( planted_partition_graph, output_folder='/app/host/networkx_generators', output_name='planted_partition_graph' )

def draw_gaussian_random_partition_graph():
    n = 20
    s = 3
    v = 2
    p_in = 0.5
    p_out = 0.5
    gaussian_random_partition_graph = nx.generators.gaussian_random_partition_graph( n, s, v, p_in, p_out )
    draw( gaussian_random_partition_graph, output_folder='/app/host/networkx_generators', output_name='gaussian_random_partition_graph' )

def draw_ring_of_cliques():
    num_cliques = 8
    clique_size = 4
    ring_of_cliques = nx.generators.ring_of_cliques( num_cliques, clique_size )
    draw( ring_of_cliques, output_folder='/app/host/networkx_generators', output_name='ring_of_cliques' )

def draw_stochastic_block_model():
    sizes = [75, 75, 300]
    probs = [[0.25, 0.05, 0.02],
             [0.05, 0.35, 0.07],
             [0.02, 0.07, 0.40]]
    stochastic_block_model = nx.generators.stochastic_block_model( sizes, probs )
    draw( stochastic_block_model, output_folder='/app/host/networkx_generators', output_name='stochastic_block_model' )

def draw_windmill_graph():
    n = 10
    k = 3
    windmill_graph = nx.generators.windmill_graph( n, k )
    draw( windmill_graph, output_folder='/app/host/networkx_generators', output_name='windmill_graph' )

def draw_spectral_graph_forge():
    return # ?
    spectral_graph_forge = nx.generators.spectral_graph_forge( G, alpha )
    draw( spectral_graph_forge, output_folder='/app/host/networkx_generators', output_name='spectral_graph_forge' )

def draw_random_tree():
    n = 5
    random_tree = nx.generators.random_tree( n )
    draw( random_tree, output_folder='/app/host/networkx_generators', output_name='random_tree' )

def draw_prefix_tree():
    return
    paths = ['ab', 'abs', 'ad']
    prefix_tree = nx.generators.prefix_tree( paths )
    draw( prefix_tree, output_folder='/app/host/networkx_generators', output_name='prefix_tree' )

def draw_nonisomorphic_trees():
    return
    order = 5
    nonisomorphic_trees = nx.generators.nonisomorphic_trees( order )
    draw( nonisomorphic_trees, output_folder='/app/host/networkx_generators', output_name='nonisomorphic_trees' )

def draw_triad_graph():
    triad_name = '120C'
    triad_graph = nx.generators.triad_graph( triad_name )
    draw( triad_graph, output_folder='/app/host/networkx_generators', output_name='triad_graph' )

def draw_joint_degree_graph():
    joint_degrees = {1: {4: 1},
                         2: {2: 2, 3: 2, 4: 2},
                         3: {2: 2, 4: 1},
                         4: {1: 1, 2: 2, 3: 1}}
    joint_degree_graph = nx.generators.joint_degree_graph( joint_degrees )
    draw( joint_degree_graph, output_folder='/app/host/networkx_generators', output_name='joint_degree_graph' )

def draw_mycielskian():
    return # ?
    mycielskian = nx.generators.mycielskian( G )
    draw( mycielskian, output_folder='/app/host/networkx_generators', output_name='mycielskian' )

def draw_mycielski_graph():
    n = 5
    mycielski_graph = nx.generators.mycielski_graph( n )
    draw( mycielski_graph, output_folder='/app/host/networkx_generators', output_name='mycielski_graph' )

def draw_all():
    draw_graph_atlas()
    draw_graph_atlas_g()
    draw_balanced_tree()
    draw_barbell_graph()
    draw_complete_graph()
    draw_complete_multipartite_graph()
    draw_circular_ladder_graph()
    draw_circulant_graph()
    draw_cycle_graph()
    draw_dorogovtsev_goltsev_mendes_graph()
    draw_empty_graph()
    draw_full_rary_tree()
    draw_ladder_graph()
    draw_lollipop_graph()
    draw_null_graph()
    draw_path_graph()
    draw_star_graph()
    draw_trivial_graph()
    draw_turan_graph()
    draw_wheel_graph()
    draw_margulis_gabber_galil_graph()
    draw_chordal_cycle_graph()
    draw_grid_2d_graph()
    draw_grid_graph()
    draw_hexagonal_lattice_graph()
    draw_hypercube_graph()
    draw_triangular_lattice_graph()
    draw_make_small_graph()
    draw_LCF_graph()
    draw_bull_graph()
    draw_chvatal_graph()
    draw_cubical_graph()
    draw_desargues_graph()
    draw_diamond_graph()
    draw_dodecahedral_graph()
    draw_frucht_graph()
    draw_heawood_graph()
    draw_hoffman_singleton_graph()
    draw_house_graph()
    draw_house_x_graph()
    draw_icosahedral_graph()
    draw_krackhardt_kite_graph()
    draw_moebius_kantor_graph()
    draw_octahedral_graph()
    draw_pappus_graph()
    draw_petersen_graph()
    draw_sedgewick_maze_graph()
    draw_tetrahedral_graph()
    draw_truncated_cube_graph()
    draw_truncated_tetrahedron_graph()
    draw_tutte_graph()
    draw_fast_gnp_random_graph()
    draw_gnp_random_graph()
    draw_dense_gnm_random_graph()
    draw_gnm_random_graph()
    draw_erdos_renyi_graph()
    draw_binomial_graph()
    draw_newman_watts_strogatz_graph()
    draw_watts_strogatz_graph()
    draw_connected_watts_strogatz_graph()
    draw_random_regular_graph()
    draw_barabasi_albert_graph()
    draw_extended_barabasi_albert_graph()
    draw_powerlaw_cluster_graph()
    draw_random_kernel_graph()
    draw_random_lobster()
    draw_random_shell_graph()
    draw_random_powerlaw_tree()
    draw_random_powerlaw_tree_sequence()
    draw_random_kernel_graph()
    draw_duplication_divergence_graph()
    draw_partial_duplication_graph()
    draw_configuration_model()
    draw_directed_configuration_model()
    draw_expected_degree_graph()
    draw_havel_hakimi_graph()
    draw_directed_havel_hakimi_graph()
    draw_degree_sequence_tree()
    draw_random_degree_sequence_graph()
    draw_random_clustered_graph()
    draw_gn_graph()
    draw_gnr_graph()
    draw_gnc_graph()
    draw_random_k_out_graph()
    draw_scale_free_graph()
    draw_random_geometric_graph()
    draw_soft_random_geometric_graph()
    draw_geographical_threshold_graph()
    draw_waxman_graph()
    draw_navigable_small_world_graph()
    draw_thresholded_random_geometric_graph()
    draw_line_graph()
    draw_inverse_line_graph()
    draw_ego_graph()
    draw_stochastic_graph()
    draw_uniform_random_intersection_graph()
    draw_k_random_intersection_graph()
    draw_general_random_intersection_graph()
    draw_karate_club_graph()
    draw_davis_southern_women_graph()
    draw_florentine_families_graph()
    draw_caveman_graph()
    draw_connected_caveman_graph()
    draw_relaxed_caveman_graph()
    draw_random_partition_graph()
    draw_planted_partition_graph()
    draw_gaussian_random_partition_graph()
    draw_ring_of_cliques()
    draw_stochastic_block_model()
    draw_windmill_graph()
    draw_spectral_graph_forge()
    draw_random_tree()
    draw_prefix_tree()
    draw_nonisomorphic_trees()
    draw_triad_graph()
    draw_joint_degree_graph()
    draw_mycielskian()
    draw_mycielski_graph()
