#ifndef LDB_HNSW_BUILD_H
#define LDB_HNSW_BUILD_H

#include <access/genam.h>
#include <common/relpath.h>
#include <nodes/execnodes.h>
#include <utils/relcache.h>

#include "hnsw.h"
#include "lib_interface.h"
#include "usearch.h"

typedef struct HnswBuildState
{
    /* Info */
    Relation   heap;
    Relation   index;
    IndexInfo *indexInfo;

    /* Settings */
    int            dimensions;
    HnswColumnType columnType;
    char          *index_file_path;
    bool           postponed;

    /* Statistics */
    double tuples_indexed;
    double reltuples;

    /* hnsw */
    hnsw_t          hnsw;
    usearch_index_t usearch_index;

    /* Memory */
    MemoryContext tmpCtx;
} HnswBuildState;

IndexBuildResult *ldb_ambuild(Relation heap, Relation index, IndexInfo *indexInfo);
void              ldb_ambuildunlogged(Relation index);
int               GetHnswIndexDimensions(Relation index, IndexInfo *indexInfo);
void              CheckHnswIndexDimensions(Relation index, Datum arrayDatum, int dimensions);
void BuildIndex(Relation heap, Relation index, IndexInfo *indexInfo, HnswBuildState *buildstate, ForkNumber forkNum);
// todo: does this render my check unnecessary
#endif  // LDB_HNSW_BUILD_H
