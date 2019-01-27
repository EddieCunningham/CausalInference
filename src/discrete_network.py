import networkx as nx
import numpy as np
from .markov_network import MarkovNetwork
import itertools
from host.src.clique import DiscreteClique
import string
from functools import reduce
from collections import namedtuple, Iterable
from .util import log_einsum, log_einsum_path

class DiscreteNetwork( MarkovNetwork ):

    PotentialComputationInstructions = namedtuple( 'PotentialComputationInstructions', [ 'node_to_eliminate',
                                                                                         'nodes',
                                                                                         'other_potentials',
                                                                                         'contract',
                                                                                         'keep_full_factor' ] )
    ComputationInstruction = namedtuple( 'ComputationInstruction', [ 'message',
                                                                     'nodes_in_potential',
                                                                     'incoming_messages',
                                                                     'contract' ] )
    BatchedInstruction = namedtuple( 'BatchedInstruction', [ 'batch_contract', 'instruction' ] )
    MessageSize = namedtuple( 'MessageSize', [ 'nodes', 'state_sizes' ] )

    def set_potentials( self, potentials ):
        """ Set the clique potentials.  THESE MUST BE IN LOG SPACE

        Args:
            potentials - A dictionary indexed by a sorted tuple of nodes with the potentials as values

        Returns:
            None
        """
        assert isinstance( potentials, dict )
        for nodes, value in potentials.items():
            assert tuple( sorted( nodes ) ) == nodes
            assert isinstance( value, np.ndarray )
        self.potentials          = potentials
        self.evidence_potentials = potentials

    def set_state_sizes( self, state_sizes ):
        """ Set the state sizes for all of the nodes

        Args:
            state_sizes - The state sizes for each node

        Returns:
            None
        """
        self.state_sizes = state_sizes

    def build_clique( self, nodes ):
        """ Build a clique over nodes.  Subclasses of MarkovNetwork should override this

        Args:
            nodes - A list of nodes to build a clique over

        Returns:
            clique - The clique object
        """
        state_sizes = [ self.state_sizes[node] for node in nodes ]
        return DiscreteClique( nodes, state_sizes )

    def junction_tree( self, *args, **kwargs ):
        assert hasattr( self, 'state_sizes' ), 'Need to specify the state sizes for the nodes'
        return super().junction_tree( *args, **kwargs )

    def add_evidence( self, nodes, data ):
        """ Add evidence to the graph.  If there is more than one piece of data for a node,
            then just multiply the potentials together (add in log space).

        Args:
            nodes - A list of nodes
            data  - A list of data for each node.  This should be all of the evidence
                    observed for this node.

        Returns:
            None
        """
        if( not isinstance( nodes, Iterable ) ):
            nodes = [ nodes ]
        if( not isinstance( data, Iterable ) ):
            data = [ data ]
        assert len( nodes ) == len( data )

        # Make sure possible states is an array
        for possible_states in data:
            assert isinstance( possible_states, Iterable )

        # Reset the evidence potentials
        for node in nodes:
            for clique in self.potentials.keys():
                if( node in clique ):
                    self.evidence_potentials[clique] = np.zeros( self.evidence_potentials[clique].shape )

        # Find the impossible indices for each clique
        clique_impossible_indices = {}
        for node, possible_states in zip( nodes, data ):

            # Find the potentials that node is a part of
            for clique, value in self.potentials.items():
                if( node in clique ):

                    # Find the impossible states
                    axis = clique.index( node )
                    state_size = self.state_sizes[node]

                    # For every observation, update the evidence
                    impossible_states = np.setdiff1d( np.arange( state_size ), possible_states )

                    # Create an indexer to select all of the impossible states
                    impossible_indices = [ slice( 0, size ) if i != axis else impossible_states for i, size in enumerate( self.potentials[clique].shape ) ]

                    if( clique not in clique_impossible_indices ):
                        clique_impossible_indices[clique] = []

                    clique_impossible_indices[clique].append( impossible_indices )

        # Update the evidence potentials
        for clique, all_impossible_indices in clique_impossible_indices.items():

            # Create the zero'd out potential
            potential = self.potentials[clique].copy()
            for impossible_indices in all_impossible_indices:
                potential[impossible_indices] = np.NINF

            # Update the evidence potential
            self.evidence_potentials[clique] = potential

    def parse_max_clique_potential_instructions( self, max_clique_potential_instructions, target_max_cliques ):
        """ Assign actual computations to the instructions

        Args:
            max_clique_potential_instructions - The instructions on how to
                            create the max_clique potentials in the junction tree

        Returns:
            supernode_potential_instructions - How to get the supernode potentials
        """
        alphabet = string.ascii_letters

        supernode_potential_instructions = []

        # Create the factors
        factors = self.evidence_potentials.copy()

        # Figure out the clique potential instructions
        for node_to_eliminate, elimination_nodes, factors_to_combine in max_clique_potential_instructions:

            # Assign a letter for each unique node
            unique_nodes = sorted( list( set( reduce( lambda x, y : x + y, factors_to_combine ) ) ) )
            node_map = dict( [ ( node, alphabet[i] ) for i, node in enumerate( unique_nodes ) ] )

            # Create the einsum contraction
            contract = ','.join( [ ''.join( [ node_map[node] for node in node_list ] ) for node_list in factors_to_combine ] )

            # See what we should keep
            expanded_contract = contract + '->' + ''.join( [ node_map[ node ] for node in unique_nodes ] )
            contract += '->' + ''.join( [ node_map[ node ] for node in unique_nodes if node != node_to_eliminate ] )

            new_factor_nodes = tuple( sorted( list( set( elimination_nodes ) - set( [ node_to_eliminate ] ) ) ) )
            factors[new_factor_nodes] = contract

            # See if this is a target max clique or if it is a temporary computation
            keep_full_factor = elimination_nodes in target_max_cliques

            instruction = DiscreteNetwork.PotentialComputationInstructions( node_to_eliminate, new_factor_nodes, factors_to_combine, contract, keep_full_factor )
            supernode_potential_instructions.append( instruction )

        return supernode_potential_instructions

    def parse_inference_instructions( self, message_instructions ):
        """ Assign actual computations to the instructions

        Args:
            message_instructions - The instructions on how to perform inference

        Returns:
            messages                 - Holds the nodes that the separator encapsulates and the state sizes for the nodes
            computation_instructions - Actually what to do
        """

        alphabet = string.ascii_letters

        computation_instructions = []
        messages = {}
        message_separator_nodes = {}

        # Figure out the message passing computations
        for instruction_batch in message_instructions:

            # Process each instruction in this batch
            batch = []
            for message, incoming_messages, separator_nodes in instruction_batch:

                origin_nodes, destination_nodes = message
                nodes_in_potential = origin_nodes

                message_separator_nodes[ message ] = separator_nodes

                # Collect the nodes.
                incoming_separators = [ list( message_separator_nodes[ message ] ) for message in incoming_messages ]
                node_lists = [ list( nodes_in_potential ), *incoming_separators ]

                # Assign a letter for each unique node
                unique_nodes = sorted( list( set( reduce( lambda x, y : x + y, node_lists ) ) ) )
                node_map = dict( [ ( node, alphabet[i] ) for i, node in enumerate( unique_nodes ) ] )

                # Create the einsum contraction
                contract = ','.join( [ ''.join( [ node_map[node] for node in node_list ] ) for node_list in node_lists ] )

                # See what we should keep
                contract += '->' + ''.join( [ node_map[ node ] for node in separator_nodes ] )

                # Store the size of this message and the nodes it is over
                messages[message] = DiscreteNetwork.MessageSize( separator_nodes, [ self.state_sizes[node] for node in separator_nodes ] )

                # Append to the total list of computations
                instruction = DiscreteNetwork.ComputationInstruction( message, nodes_in_potential, incoming_messages, contract )

                batch.append( instruction )

            # Computations can only be batched if they have the same contract and if their contract has the same sized dimensions
            group_by_contract_then_size = {}
            for instruction in batch:
                message, nodes_in_potential, incoming_messages, contract = instruction

                # Find the sizes for the contract
                potential_sizes = [ self.state_sizes[node] for node in nodes_in_potential ]
                incoming_sizes = [ messages[in_message].state_sizes for in_message in incoming_messages ]
                contract_sizes = tuple( reduce( lambda x,y:x+y, [ potential_sizes, *incoming_sizes ] ) )

                if( contract not in group_by_contract_then_size ):
                    group_by_contract_then_size[contract] = {}

                if( contract_sizes not in group_by_contract_then_size[contract] ):
                    group_by_contract_then_size[contract][contract_sizes] = []

                # Add the current instruction to the group
                group_by_contract_then_size[contract][contract_sizes].append( instruction )

            # Group the messages together in batches and create the new contract
            batch_letter = alphabet[-1]
            batched_instructions = []
            for contract, sizes_and_messages in group_by_contract_then_size.items():
                for sizes, instructions in sizes_and_messages.items():

                    new_contract = ( ',' + batch_letter ).join( contract.split( ',' ) )
                    new_contract = batch_letter + ( '->' + batch_letter ).join( new_contract.split( '->' ) )
                    batched_instruction = DiscreteNetwork.BatchedInstruction( new_contract, instructions )

                    batched_instructions.append( batched_instruction )

            computation_instructions.extend( batched_instructions )

        return messages, computation_instructions

    def evaluate_instruction_complexity( self,
                                         supernode_potential_instructions,
                                         messages,
                                         computation_instructions,
                                         gpu_support=True ):
        """ Determing how fast it will be to do computations on the given instructions.

        Args:
            supernode_potential_instructions - How to get the supernode potentials
            messages                         - Holds the nodes that the separator encapsulates and the state sizes for the nodes
            computation_instructions         - Actually what to do
            gpu_support                      - Whether or not we intend to do inference on a gpu

        Returns:
            A number that roughly represents the computational complexity
        """
        def contraction_complexity( contract, contraction_list ):
            if( gpu_support == True ):
                # Not using a huge dataset, so the computation its self on a gpu
                # is probably going to be the same for each contraction regardless
                # of the size
                return len( contraction_list )

            # Count the number of computations done.  This involves the state sizes
            assert 0, 'Not implemented yet'

        total_complexity = 0

        # Build the clique potentials
        potentials = dict( [ ( key, np.empty( potential.shape ) ) for key, potential in self.evidence_potentials.items() ] )

        for node_to_eliminate, nodes, other_potentials, contract, keep in supernode_potential_instructions:

            # Create the new factor
            if( keep ):
                all_nodes = tuple( sorted( [ node_to_eliminate ] + list( nodes ) ) )

                # Create the full contration
                new_contract, reduced_contract = contract.split( '->' )
                expanded_contract = ''.join( sorted( list( set( new_contract ) ) ) ).replace( ',', '' )
                new_contract += '->' + expanded_contract
                new_contract_reduced = expanded_contract + '->' + reduced_contract

                # Find the contractions
                potentials[all_nodes], cont_list1 = log_einsum_path( new_contract, *[ potentials[other_nodes] for other_nodes in other_potentials ] )
                potentials[nodes], cont_list2 = log_einsum_path( new_contract_reduced, potentials[all_nodes] )

                # Update the total complexity
                total_complexity += contraction_complexity( new_contract, cont_list1 )
                total_complexity += contraction_complexity( new_contract_reduced, cont_list2 )
            else:
                potentials[nodes], cont_list = log_einsum_path( contract, *[ potentials[other_nodes] for other_nodes in other_potentials ] )
                total_complexity += contraction_complexity( contract, cont_list )

        # Perform the inference computations
        message_objects = dict( [ ( message, np.empty( s.state_sizes ) ) for message, s in messages.items() ] )
        for batched_contract, instructions in computation_instructions:

            # Find the potential objects
            all_potentials = np.array( [ potentials[instruction.nodes_in_potential] for instruction in instructions ] )

            # Batch the incoming message objects
            incoming_message_batches = zip( *[ instruction.incoming_messages for instruction in instructions ] )
            incoming_message_tensors = [ np.array( [ message_objects[message] for message in message_batch ] ) for message_batch in incoming_message_batches ]

            # Do the actual computation
            _, cont_list = log_einsum_path( batched_contract, all_potentials, *incoming_message_tensors )
            total_complexity += contraction_complexity( batched_contract, cont_list )

        return total_complexity

    def generate_contractions( self,
                               supernode_potential_instructions,
                               messages,
                               computation_instructions ):
        """ Pre-fetch the contractions for the computations.  For graphs with a lot of terms in the contract
            and smaller state sizes, finding the the contractions can be expensive compared to the actual computation.

        Args:
            supernode_potential_instructions - How to get the supernode potentials
            messages                         - Holds the nodes that the separator encapsulates and the state sizes for the nodes
            computation_instructions         - Actually what to do

        Returns:
            contraction_lists - The order of contractions to perform at each log_einsum step
        """

        contraction_lists = []

        # Build the clique potentials
        potentials = dict( [ ( key, np.empty( potential.shape ) ) for key, potential in self.evidence_potentials.items() ] )

        for node_to_eliminate, nodes, other_potentials, contract, keep in supernode_potential_instructions:

            # Create the new factor
            if( keep ):
                all_nodes = tuple( sorted( [ node_to_eliminate ] + list( nodes ) ) )
                new_contract, reduced_contract = contract.split( '->' )
                expanded_contract = ''.join( sorted( list( set( new_contract ) ) ) ).replace( ',', '' )
                new_contract += '->' + expanded_contract
                potentials[all_nodes], cont_list1 = log_einsum_path( new_contract, *[ potentials[other_nodes] for other_nodes in other_potentials ] )
                potentials[nodes], cont_list2 = log_einsum_path( expanded_contract + '->' + reduced_contract, potentials[all_nodes] )
                contraction_lists.extend( [ cont_list1, cont_list2 ] )
            else:
                potentials[nodes], cont_list = log_einsum_path( contract, *[ potentials[other_nodes] for other_nodes in other_potentials ] )
                contraction_lists.append( cont_list )

        # Perform the inference computations
        message_objects = dict( [ ( message, np.empty( s.state_sizes ) ) for message, s in messages.items() ] )
        for batched_contract, instructions in computation_instructions:

            # Find the potential objects
            all_potentials = np.array( [ potentials[instruction.nodes_in_potential] for instruction in instructions ] )

            # Batch the incoming message objects
            incoming_message_batches = zip( *[ instruction.incoming_messages for instruction in instructions ] )
            incoming_message_tensors = [ np.array( [ message_objects[message] for message in message_batch ] ) for message_batch in incoming_message_batches ]

            # Do the actual computation
            message, cont_list = log_einsum_path( batched_contract, all_potentials, *incoming_message_tensors )
            contraction_lists.append( cont_list )

        return contraction_lists

    def perform_message_passing( self,
                                 supernode_potential_instructions,
                                 messages,
                                 computation_instructions,
                                 contraction_lists ):
        """ Actually do the computations

        Args:
            supernode_potential_instructions - How to get the supernode potentials
            messages                         - Holds the nodes that the separator encapsulates and the state sizes for the nodes
            computation_instructions         - Actually what to do
            contraction_lists                - The order of contractions to perform at each log_einsum step

        Returns:
            marginals - The node marginals
        """

        alphabet = string.ascii_letters
        contraction_iter = iter( contraction_lists )

        # Build the clique potentials
        potentials = self.evidence_potentials.copy()

        for node_to_eliminate, nodes, other_potentials, contract, keep in supernode_potential_instructions:

            # Create the new factor
            if( keep ):
                all_nodes = tuple( sorted( [ node_to_eliminate ] + list( nodes ) ) )
                new_contract, reduced_contract = contract.split( '->' )
                expanded_contract = ''.join( sorted( list( set( new_contract ) ) ) ).replace( ',', '' )
                new_contract += '->' + expanded_contract
                potentials[all_nodes] = log_einsum( new_contract, *[ potentials[other_nodes] for other_nodes in other_potentials ], contraction_list=next( contraction_iter ) )
                potentials[nodes] = log_einsum( expanded_contract + '->' + reduced_contract, potentials[all_nodes], contraction_list=next( contraction_iter ) )
            else:
                potentials[nodes] = log_einsum( contract, *[ potentials[other_nodes] for other_nodes in other_potentials ], contraction_list=next( contraction_iter ) )

        # Perform the inference computations
        message_objects = dict( [ ( message, np.empty( s.state_sizes ) ) for message, s in messages.items() ] )
        for batched_contract, instructions in computation_instructions:

            # Find the potential objects
            all_potentials = np.array( [ potentials[instruction.nodes_in_potential] for instruction in instructions ] )

            # Batch the incoming message objects
            incoming_message_batches = zip( *[ instruction.incoming_messages for instruction in instructions ] )
            incoming_message_tensors = [ np.array( [ message_objects[message] for message in message_batch ] ) for message_batch in incoming_message_batches ]

            # Do the actual computation
            message = log_einsum( batched_contract, all_potentials, *incoming_message_tensors, contraction_list=next( contraction_iter ) )

            # Copy the results back to the appropriate message objects
            for i, instruction in enumerate( instructions ):
                message_objects[instruction.message] = message[i]

        # Collect the clique potentials
        marginals = {}
        for ( nodes, _ ), data in filter( lambda x: x[0][0]==x[0][1], message_objects.items() ):

            # See which marginals are needed within this cluster
            needed = [ node for node in nodes if node not in marginals ]

            # print( nodes )

            potential = potentials[nodes]

            for node in needed:

                contract = ''.join( [ alphabet[i] for i, _node in enumerate( nodes ) ] )
                contract += '->' + ''.join( [ alphabet[i] for i, _node in enumerate( nodes ) if _node == node ] )
                marginals[node] = log_einsum( contract, data )

                contract = ''.join( [ alphabet[i] for i, _node in enumerate( nodes ) ] ) + '->'
                total = log_einsum( contract, data )
                print( node, total, marginals[node] - total )

        return marginals

    def find_best_elimination_order( self, n_iters=100, gpu_support=True ):
        """ Find the elimination order that gives the most efficient computation instructions

        Args:
            gpu_support - Whether or not we intend to do inference on a gpu
            n_iters     - The number of iterations to perform

        Returns:
            order - The best elimination order
        """

        best_order = None
        lowest_complexity = 999999999999999999

        for _ in range( n_iters ):

            # Run variable elimination
            order, max_clique_potential_instructions, maximal_cliques = self.variable_elimination( clique_factorization=self.evidence_potentials,
                                                                                                   return_maximal_cliques=True,
                                                                                                   draw=False )

            # Create the junction tree and the computation instructions
            junction_tree = self.junction_tree( maximal_cliques )
            instructions = junction_tree.shafer_shenoy_inference_instructions()

            # Generate the instructions to do inference
            supernode_potential_instructions = self.parse_max_clique_potential_instructions( max_clique_potential_instructions, junction_tree.nodes )
            separators, computation_instructions = self.parse_inference_instructions( instructions )

            # See how good this order is
            instruction_complexity = self.evaluate_instruction_complexity( supernode_potential_instructions, separators, computation_instructions, gpu_support=gpu_support )

            if( ( best_order is None ) or ( instruction_complexity < lowest_complexity ) ):
                best_order = order
                lowest_complexity = instruction_complexity

        return order, lowest_complexity

    def get_computation_instructions( self, order=None ):
        """ Get the computation instructions for an elimination order

        Args:
            order - The elimination order.  If is None, then will find (a probably suboptimal) one.

        Returns:
            supernode_potential_instructions - The instructions to compute the supernode potentials
                                               (DiscreteNetwork.PotentialComputationInstructions)
            separators                       - Keys for the separators (what we compute during inference)
                                               and their state sizes (DiscreteNetwork.MessageSize)
            computation_instructions         - The actual computations (DiscreteNetwork.ComputationInstruction)
            contraction_lists                - The contractions for each log_einsum call in computation_instructions
        """
        # Run variable elimination
        _, max_clique_potential_instructions, maximal_cliques = self.variable_elimination( clique_factorization=self.evidence_potentials,
                                                                                           order=order,
                                                                                           return_maximal_cliques=True,
                                                                                           draw=False )

        # Create the junction tree and the computation instructions
        junction_tree = self.junction_tree( maximal_cliques )
        instructions = junction_tree.shafer_shenoy_inference_instructions()

        # Generate the instructions to do inference
        supernode_potential_instructions = self.parse_max_clique_potential_instructions( max_clique_potential_instructions, junction_tree.nodes )
        separators, computation_instructions = self.parse_inference_instructions( instructions )

        # Generate the einsum contractions
        contraction_lists = self.generate_contractions( supernode_potential_instructions, separators, computation_instructions )

        return supernode_potential_instructions, separators, computation_instructions, contraction_lists