#include <postgres.h>

#include "insert.h"

#include <access/generic_xlog.h>
#include <assert.h>
#if PG_VERSION_NUM >= 150000
#include <common/pg_prng.h>
#endif
#include <float.h>
#include <math.h>
#include <miscadmin.h>
#include <storage/bufmgr.h>
#include <utils/array.h>
#include <utils/rel.h>
#include <utils/relcache.h>

#include "build.h"
#include "external_index.h"
#include "hnsw.h"
#include "options.h"
#include "retriever.h"
#include "usearch.h"
#include "utils.h"
#include "vector.h"

/*
 * Generate a random level for a new externally stored vector
 */
static uint32 hnsw_generate_new_level(size_t connectivity)
{
    double inverse_log_connectivity = 1.0 / log((double)connectivity);
    // note: RNG is initialized (via srandom or via an updated mechanism) in postmaster.c
    // we want rand_num to be in range [0.0, 1.0)
#if PG_VERSION_NUM >= 150000
    double rand_num = pg_prng_double(&pg_global_prng_state);
#else
    double rand_num = (double)random() / (double)MAX_RANDOM_VALUE;
#endif
    double level = -1 * log(rand_num) * inverse_log_connectivity;
    return (uint32)level;
}

/*
 * Insert a tuple into the index
 */
bool ldb_aminsert(Relation         index,
                  Datum           *values,
                  bool            *isnull,
                  ItemPointer      heap_tid,
                  Relation         heap,
                  IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
                  ,
                  bool indexUnchanged
#endif
                  ,
                  IndexInfo *indexInfo)
{
    MemoryContext          oldCtx;
    MemoryContext          insertCtx;
    Datum                  datum;
    usearch_index_t        uidx;
    usearch_error_t        error = NULL;
    usearch_metadata_t     meta;
    BlockNumber            HEADER_BLOCK = 0;
    Buffer                 hdr_buf;
    Page                   hdr_page;
    HnswIndexHeaderPage   *hdr;
    GenericXLogState      *state;
    uint32                 new_tuple_id;
    HnswIndexTuple        *new_tuple;
    HnswColumnType         column_type;
    usearch_init_options_t opts = {0};
    LDB_UNUSED(heap);
#if PG_VERSION_NUM >= 140000
    LDB_UNUSED(indexUnchanged);
#endif
    
    HnswInsertState *insertstate = palloc0(sizeof(HnswInsertState));

    if(checkUnique != UNIQUE_CHECK_NO) {
        elog(ERROR, "unique constraints on hnsw vector indexes not supported");
    }

    // q:: what are realistic cases where the vector fields would be null, other than the case
    // todo:: where the column is created and not all embeddings have been computed yet?
    // perhaps we should add a null bitmap to the index and support WHERE queries for exact null lookups?
    if(isnull[ 0 ]) {
        return false;
    }
    // todo:: there is room for optimization for when indexUnchanged is true
    //elog(INFO, "RUNNING AMINSERT!");

    // TODO: what if there are concurrent inserts? can that result in issues with creating this index, which we've
    // postponed, to here?
    // bool postponed = (indexInfo->ii_AmCache != NULL);
    datum = PointerGetDatum(PG_DETOAST_DATUM(values[ 0 ]));
    column_type = GetIndexColumnType(index);
    //elog(INFO, "HEAP number of blocks: %d", RelationGetNumberOfBlocks(heap));

    bool postponed = RelationGetNumberOfBlocks(index) == 0;

    //elog(INFO, "INSERT: ii_AmCache address: %p\n", indexInfo->ii_AmCache);
    //elog(INFO, "Num Blocks in insert: %d", RelationGetNumberOfBlocks(index));

    if(postponed) {
        // int ndims = GetHnswIndexDimensions(index, indexInfo);
        int ndims = DatumGetLength(datum, column_type);
        // TODO: test this ldb_HnswGetDim call... not sure if options will carry over
        int index_ndims = ldb_HnswGetDim(index);
        // TODO: check if above works... we can then only do this postponed index build stuff if no dimension was
        // specified during index declaration
        // int index_ndims = -1;
        //elog(INFO, "Vector dimension specified during index creation: %d", index_ndims);
        //elog(INFO, "ndims is : %d", ndims);
        // elog(INFO, "From aminsert, our index_ndims is: %d", index_ndims);

        if(index_ndims >= 1 && index_ndims != ndims) {
            elog(ERROR,
                 "Vector dimension %d of inserted vector does not match vector dimension %d specified during index "
                 "creation.",
                 ndims,
                 index_ndims);
            return false;
        }

        if(ndims < 1) {
            // error is printed in GetHNSWIndexDimensions above
            elog(ERROR, "ndims < 1 in insert!");
            return false;
        }

        if(ndims > HNSW_MAX_DIM) {
            elog(ERROR,
                 "Vector dimension %d of inserted vector is too large. "
                 "LanternDB currently supports up to %ddim vectors",
                 ndims,
                 HNSW_MAX_DIM);
            return false;
        }

        // construct index with ndims now
        HnswBuildState buildstate;
        buildstate.postponed = true;
        buildstate.dimensions = ndims;

        // dont think we need this context switching stuff
        // MemoryContext prevCtx;
        // prevCtx = CurrentMemoryContext;

        //elog(INFO, "About to call build index for the second time");
        BuildIndex(heap, index, indexInfo, &buildstate, MAIN_FORKNUM);
        //elog(INFO, "Finished building Index!");

        // Building the index already inserted this vector since it was written to the heap prior to this AM method
        // being called, so we can return to avoid inserting twice
        return false;

        // MemoryContextSwitchTo(prevCtx);
        // elog(INFO, "Finished switching back to prevCtx!");
    }

    insertCtx = AllocSetContextCreate(CurrentMemoryContext, "LanternInsertContext", ALLOCSET_DEFAULT_SIZES);
    oldCtx = MemoryContextSwitchTo(insertCtx);

    state = GenericXLogStart(index);

    //  read index header page to know how many pages are already inserted
    hdr_buf = ReadBufferExtended(index, MAIN_FORKNUM, HEADER_BLOCK, RBM_NORMAL, NULL);
    LockBuffer(hdr_buf, BUFFER_LOCK_EXCLUSIVE);
    // header page MUST be under WAL since PrepareIndexTuple will update it
    hdr_page = GenericXLogRegisterBuffer(state, hdr_buf, LDB_GENERIC_XLOG_DELTA_IMAGE);
    hdr = (HnswIndexHeaderPage *)PageGetContents(hdr_page);
    assert(hdr->magicNumber == LDB_WAL_MAGIC_NUMBER);

    //opts.dimensions = GetHnswIndexDimensions(index, indexInfo);
    opts.dimensions = hdr->vector_dim;

    //opts.dimensions = DatumGetLength(datum, column_type);
    //elog(INFO, "opts.dimensions in insert: %d", (int)opts.dimensions);
    CheckHnswIndexDimensions(index, values[ 0 ], opts.dimensions);
    PopulateUsearchOpts(index, &opts);
    opts.retriever_ctx = ldb_wal_retriever_area_init(index, hdr);
    opts.retriever = ldb_wal_index_node_retriever;
    opts.retriever_mut = ldb_wal_index_node_retriever_mut;

    // todo:: do usearch init in indexInfo->ii_AmCache
    uidx = usearch_init(&opts, &error);
    if(uidx == NULL) {
        elog(ERROR, "unable to initialize usearch");
    }
    assert(!error);

    assert(usearch_size(uidx, &error) == 0);
    assert(!error);

    usearch_view_mem_lazy(uidx, hdr->usearch_header, &error);
    assert(!error);

    insertstate->uidx = uidx;
    insertstate->retriever_ctx = opts.retriever_ctx;
    insertstate->columnType = column_type;

    hdr_page = NULL;

    meta = usearch_metadata(uidx, &error);
    assert(!error);

    void *vector = DatumGetSizedArray(datum, insertstate->columnType, opts.dimensions);

#if LANTERNDB_COPYNODES
    // currently not fully ported to the latest changes
    assert(false);
#else
    assert(dlist_is_empty(&insertstate->retriever_ctx->takenbuffers));
#endif

    assert(hdr->magicNumber == LDB_WAL_MAGIC_NUMBER);
    ldb_dlog("Insert: at start num vectors is %d", hdr->num_vectors);

    CheckMem(work_mem,
             index,
             uidx,
             hdr->num_vectors,
             "index size exceeded work_mem during insert, consider increasing work_mem");

    usearch_reserve(uidx, hdr->num_vectors + 1, &error);
    uint32 level = hnsw_generate_new_level(meta.connectivity);
    if(error != NULL) {
        elog(ERROR, "usearch newnode error: %s", error);
    }

    new_tuple_id = hdr->num_vectors;
    // we are adding the following pages to the Generic XLog
    // 1) the header page
    // 2) the page containing the new tuple
    // 3) (sometimes) the page that used to be last page of the index
    // 4) The blockmap page for the block in which the vector was added
    // Generic XLog supports up to 4 pages in a single commit, so we are good.
    new_tuple = PrepareIndexTuple(index, state, hdr, &meta, new_tuple_id, level, insertstate);

    usearch_add_external(
        uidx, *(unsigned long *)heap_tid, vector, new_tuple->node, usearch_scalar_f32_k, level, &error);
    if(error != NULL) {
        elog(ERROR, "usearch insert error: %s", error);
    }

    usearch_update_header(uidx, hdr->usearch_header, &error);

    ldb_wal_retriever_area_reset(insertstate->retriever_ctx, hdr);

    // we only release the header buffer AFTER inserting is finished to make sure nobody else changes the block
    // structure. todo:: critical section here can definitely be shortened
    {
        // GenericXLogFinish also calls MarkBufferDirty(buf)
        XLogRecPtr ptr = GenericXLogFinish(state);
        assert(ptr != InvalidXLogRecPtr);
        LDB_UNUSED(ptr);
    }

    extra_dirtied_release_all(insertstate->retriever_ctx->extra_dirted);

    usearch_free(insertstate->uidx, &error);
    if(error != NULL) {
        elog(ERROR, "error freeing usearch index: %s", error);
    }

    // unlock the header page
    assert(BufferIsValid(hdr_buf));
    // GenericXLogFinish already marks hdr_buf as dirty
    UnlockReleaseBuffer(hdr_buf);

    ldb_wal_retriever_area_fini(insertstate->retriever_ctx);
    pfree(insertstate);

    // q:: what happens when there is an error before ths and the switch back never happens?
    MemoryContextSwitchTo(oldCtx);

    MemoryContextDelete(insertCtx);

    // from docs at https://www.postgresql.org/docs/current/index-functions.html:
    // The function's Boolean result value is significant only when checkUnique is UNIQUE_CHECK_PARTIAL.
    // In this case a true result means the new entry is known unique, whereas false means it might be
    // non-unique (and a deferred uniqueness check must be scheduled).
    // For other cases a constant false result is recommended.
    return false;
}
