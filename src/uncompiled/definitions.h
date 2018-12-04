#ifndef __DEFINITIONS_H__
#define __DEFINITIONS_H__

namespace cython_accessors
{
    int EDGE       = 0;
    int NODE       = 1;
    int NEXT_FMLY  = 2;

    int NODE_CHDN_IDX = 0;
    int NODE_PRNT_IDX = 1;
    int N_CHILD_EDGES = 2;

    int EDGE_PRNT_IDX = 0;
    int EDGE_CHDN_IDX = 1;
    int N_PARENTS     = 2;
    int N_CHILDREN    = 3;

    int N_ROOTS         = 0;
    int N_LEAVES        = 1;
    int MAX_CHILD_EDGES = 2;
    int MAX_PARENTS     = 3;
    int MAX_CHILDREN    = 4;
}

#endif
