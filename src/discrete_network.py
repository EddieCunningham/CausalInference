import networkx as nx
import numpy as np
from .markov_network import MarkovNetwork
import itertools
from host.src.clique import DiscreteClique
import string
from functools import reduce
from collections import namedtuple
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
        """ Set the clique potentials

        Args:
            potentials - A dictionary indexed by a sorted tuple of nodes with the potentials as values

        Returns:
            None
        """
        self.potentials = potentials

    def set_state_sizes( self, state_sizes ):
        """ Set the state sizes for all of the nodes

        Args:
            nodes - A list of nodes to build a clique over

        Returns:
            clique - The clique object
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
        factors = self.potentials.copy()

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
        # TODO: Make batched computations

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
        potentials = dict( [ ( key, np.empty( potential.shape ) ) for key, potential in self.potentials.items() ] )

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
        potentials = self.potentials.copy()

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

            print()

        return marginals

    def is_consistent( self ):
        """ Check to see if inference worked.  This is done by seeing if
            pairs of nodes at each edge marginalize to the same thing.
            Because of the RIP, locally consistency implies global consistency

        Args:
            None

        Returns:
            Whether or not tree is consistent
        """
        pass
