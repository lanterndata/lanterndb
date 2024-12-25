#include <postgres.h>

#include "build.h"

#include <access/heapam.h>
#include <access/relscan.h>
#include <access/sdir.h>
#include <assert.h>
#include <catalog/index.h>
#include <catalog/namespace.h>
#include <catalog/pg_proc.h>
#include <catalog/pg_type.h>
#include <executor/executor.h>
#include <fmgr.h>
#include <funcapi.h>
#include <miscadmin.h>
#include <nodes/execnodes.h>
#include <stdint.h>
#include <storage/bufmgr.h>
#include <storage/lockdefs.h>
#include <string.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <utils/array.h>
#include <utils/builtins.h>
#include <utils/lsyscache.h>
#include <utils/memutils.h>
#include <utils/palloc.h>
#include <utils/snapmgr.h>
#include <utils/syscache.h>

#include "usearch.h"

#ifdef _WIN32
#define access _access
#else
#include <unistd.h>
#endif

#include "bench.h"
#include "external_index.h"
#include "external_index_socket.h"
#include "failure_point.h"
#include "hnsw.h"
#include "options.h"
#include "pqtable.h"
#include "retriever.h"
#include "utils.h"
#include "vector.h"

#if PG_VERSION_NUM >= 140000
#include <utils/backend_progress.h>
#elif PG_VERSION_NUM >= 120000
#include <pgstat.h>
#endif

#if PG_VERSION_NUM <= 120000
#include <access/htup_details.h>
#endif

#if PG_VERSION_NUM >= 120000
#include <access/tableam.h>
#include <commands/progress.h>
#else
#define LDB_PROGRESS_CREATEIDX_SUBPHASE     0
#define LDB_PROGRESS_CREATEIDX_TUPLES_TOTAL 0
#define LDB_PROGRESS_CREATEIDX_TUPLES_DONE  0
#endif

#if PG_VERSION_NUM >= 130000
#define CALLBACK_ITEM_POINTER ItemPointer tid
#else
#define CALLBACK_ITEM_POINTER HeapTuple hup
#endif

#if PG_VERSION_NUM >= 120000
#define UpdateProgress(index, val) pgstat_progress_update_param(index, val)
#else
#define UpdateProgress(index, val) ((void)val)
#endif

static void AddTupleToUsearchIndex(ItemPointer         tid,
                                   Datum               detoasted_vector,
                                   ldb_HnswBuildState *buildstate,
                                   Relation            index)
{
    usearch_error_t       error = NULL;
    usearch_scalar_kind_t usearch_scalar;
    uint8                 scalar_bits = 32;

    void *vector = DatumGetSizedArray(detoasted_vector, buildstate->columnType, buildstate->dimensions, false);
    switch(buildstate->columnType) {
        case REAL_ARRAY:
        case VECTOR:
            usearch_scalar = usearch_scalar_f32_k;
            break;
        case INT_ARRAY:
            // this is taken in hamming distance
            usearch_scalar = usearch_scalar_b1_k;
            scalar_bits = 1;
            break;
        default:
            pg_unreachable();
    }

    // casting tid structure to a number to be used as value in vector search
    // tid has info about disk location of this item and is 6 bytes long
    usearch_label_t label = ItemPointer2Label(tid);

    if(buildstate->external_socket) {
        // send tuple over socket if this is external indexing
        external_index_send_tuple(buildstate->external_socket, &label, vector, scalar_bits, buildstate->dimensions);

    } else if(buildstate->usearch_index != NULL) {
        size_t capacity = usearch_capacity(buildstate->usearch_index, &error);
        if(capacity == usearch_size(buildstate->usearch_index, &error)) {
            CheckMem(maintenance_work_mem,
                     index,
                     buildstate->usearch_index,
                     2 * usearch_size(buildstate->usearch_index, &error),
                     "index size exceeded maintenance_work_mem during index construction, consider increasing "
                     "maintenance_work_mem");
            usearch_reserve(buildstate->usearch_index, 2 * capacity, &error);
            assert(error == NULL);
        }

        usearch_add(buildstate->usearch_index, label, vector, usearch_scalar, &error);
    }
    assert(error == NULL);
    buildstate->tuples_indexed++;
    buildstate->reltuples++;
    UpdateProgress(PROGRESS_CREATEIDX_TUPLES_DONE, buildstate->tuples_indexed);
    UpdateProgress(PROGRESS_CREATEIDX_TUPLES_TOTAL, buildstate->reltuples);
}

/*
 * Callback for table_index_build_scan
 */
static void BuildCallback(
    Relation index, CALLBACK_ITEM_POINTER, Datum *values, bool *isnull, bool tupleIsAlive, void *state)
{
    Datum               detoasted_array;
    ldb_HnswBuildState *buildstate = (ldb_HnswBuildState *)state;
    MemoryContext       oldCtx;
    // we can later use this for some optimizations I think
    LDB_UNUSED(tupleIsAlive);

#if PG_VERSION_NUM < 130000
    ItemPointer tid = &hup->t_self;
#endif

    /* Skip nulls */
    if(isnull[ 0 ]) return;
    /* Use memory context since detoast can allocate */
    oldCtx = MemoryContextSwitchTo(buildstate->tmpCtx);

    /* Detoast once for all calls */
    detoasted_array = PointerGetDatum(PG_DETOAST_DATUM(values[ 0 ]));
    CheckHnswIndexDimensions(index, detoasted_array, buildstate->dimensions);

    // todo:: the argument values is assumed to be a real[] or vector (they have the same layout)
    // do proper type checking instead of this assumption and test int int arrays and others
    LanternBench("AddTupleToUsearch", AddTupleToUsearchIndex(tid, detoasted_array, buildstate, index));

    // free the detoasted value if we allocated for it, to avoid accumulating them during index construction
    // would be cleaner to create a memory context but not sure how much overhead
    // creating/destroying memory context in a tight loop has
    // if we did not allocate for value, then PG_DETOAST_DATUM returns the same pointer - that is what the check below
    // is
    if(detoasted_array != values[ 0 ]) {
        pfree(DatumGetPointer(detoasted_array));
    }

    /* Reset memory context */
    MemoryContextSwitchTo(oldCtx);
    MemoryContextReset(buildstate->tmpCtx);
}

static int GetArrayLengthFromExpression(Expr *expression, Relation heap, HeapTuple tuple)
{
    ExprContext    *econtext;
    ExprState      *exprstate;
    EState         *estate;
    Datum           result;
    bool            isNull;
    Oid             resultOid;
    TupleTableSlot *slot;
    TupleDesc       tupdesc = RelationGetDescr(heap);

#if PG_VERSION_NUM >= 120000
    slot = MakeSingleTupleTableSlot(tupdesc, &TTSOpsHeapTuple);
#else
    slot = MakeSingleTupleTableSlot(tupdesc);
#endif

    // Create an expression context
    econtext = CreateStandaloneExprContext();
    estate = CreateExecutorState();

    // Build the expression state for your expression
    exprstate = ExecPrepareExpr(expression, estate);

#if PG_VERSION_NUM >= 120000
    ExecStoreHeapTuple(tuple, slot, false);
#else
    ExecStoreTuple(tuple, slot, InvalidBuffer, false);
#endif
    // Set up the tuple for the expression evaluation
    econtext->ecxt_scantuple = slot;

    // Evaluate the expression for the first row
    result = ExecEvalExprSwitchContext(exprstate, econtext, &isNull);

    // Release tuple descriptor
    ReleaseTupleDesc(tupdesc);

    // Get the return type information
    get_expr_result_type((Node *)exprstate->expr, &resultOid, NULL);

    HnswColumnType columnType = GetColumnTypeFromOid(resultOid);

    if(columnType == REAL_ARRAY || columnType == INT_ARRAY) {
        ArrayType *array = DatumGetArrayTypeP(result);
        return ArrayGetNItems(ARR_NDIM(array), ARR_DIMS(array));
    } else if(columnType == VECTOR) {
        Vector *vector = DatumGetVector(result);
        return vector->dim;
    } else {
        // Check if the result is not null and is supported type
        // There is a guard in postgres that wont' allow passing
        // Anything else from the defined operator class types
        // Throwing an error like: ERROR:  data type text has no default operator class for access method "hnsw"
        // So this case will be marked as invariant
        ldb_invariant(!isNull && columnType != UNKNOWN,
                      "Expression used in CREATE INDEX statement did not result in hnsw-index compatible array");
    }

    return HNSW_DEFAULT_DIM;
}

static int GetArrayLengthFromHeap(Relation heap, int indexCol, IndexInfo *indexInfo)
{
#if PG_VERSION_NUM < 120000
    HeapScanDesc scan;
#else
    TableScanDesc scan;
#endif
    HeapTuple  tuple;
    Snapshot   snapshot;
    ArrayType *array;
    Datum      datum;
    bool       isNull;
    int        n_items = HNSW_DEFAULT_DIM;
    //
    // Get the first row off the heap
    // if it's NULL we don't infer a length. Since vectors are expected to have fixed nonzero dimension this will result
    // in an error later
    snapshot = GetTransactionSnapshot();
#if PG_VERSION_NUM < 120000
    scan = heap_beginscan(heap, snapshot, 0, NULL);
#else
    scan = heap_beginscan(heap, snapshot, 0, NULL, NULL, SO_TYPE_SEQSCAN);
#endif
    tuple = heap_getnext(scan, ForwardScanDirection);
    if(tuple == NULL) {
        heap_endscan(scan);
        return n_items;
    }

    if(indexInfo->ii_Expressions != NULL) {
        // We don't suport multicolumn indexes
        // So trying to pass multiple expressions on index creation
        // Will result an error before getting here
        ldb_invariant(indexInfo->ii_Expressions->length == 1,
                      "Index expressions can not be greater than 1 as multicolumn indexes are not supported");
        Expr *indexpr_item = lfirst(list_head(indexInfo->ii_Expressions));
        n_items = GetArrayLengthFromExpression(indexpr_item, heap, tuple);
    } else {
        // Get the indexed column out of the row and return it's dimensions
        datum = heap_getattr(tuple, indexCol, RelationGetDescr(heap), &isNull);
        if(!isNull) {
            array = DatumGetArrayTypeP(datum);
            n_items = ArrayGetNItems(ARR_NDIM(array), ARR_DIMS(array));
            // todo:: probably can pfree the array
        }
    }

    heap_endscan(scan);

    return n_items;
}

int GetHnswIndexDimensions(Relation index, IndexInfo *indexInfo)
{
    HnswColumnType columnType = GetIndexColumnType(index);

    // check if column is type of real[] or integer[]
    if(columnType == REAL_ARRAY || columnType == INT_ARRAY) {
        // The dimension in options is not set if the dimension is inferred so we need to actually check the key
        int opt_dim = ldb_HnswGetDim(index);
        if(opt_dim == HNSW_DEFAULT_DIM) {
            // If the option's still the default it needs to be updated to match what was inferred
            // todo: is there a way to do this earlier? (rd_options is null in BuildInit)
            Relation         heap;
            ldb_HnswOptions *opts;
            int              attrNum;

            assert(index->rd_index->indnatts == 1);
            attrNum = index->rd_index->indkey.values[ 0 ];
#if PG_VERSION_NUM < 120000
            heap = heap_open(index->rd_index->indrelid, AccessShareLock);
#else
            heap = table_open(index->rd_index->indrelid, AccessShareLock);
#endif
            opt_dim = GetArrayLengthFromHeap(heap, attrNum, indexInfo);
            opts = (ldb_HnswOptions *)index->rd_options;
            if(opts != NULL) {
                opts->dim = opt_dim;
            }
#if PG_VERSION_NUM < 120000
            heap_close(heap, AccessShareLock);
#else
            table_close(heap, AccessShareLock);
#endif
        }
        return opt_dim;
    } else if(columnType == VECTOR) {
        return TupleDescAttr(index->rd_att, 0)->atttypmod;
    } else {
        elog(ERROR,
             "Unsupported type"
             " LanternDB currently supports only real[] and vector types");
    }

    return -1;
}

void CheckHnswIndexDimensions(Relation index, Datum arrayDatum, int dimensions)
{
    ArrayType     *array;
    int            n_items;
    HnswColumnType indexType = GetIndexColumnType(index);

    if(indexType == REAL_ARRAY || indexType == INT_ARRAY) {
        array = DatumGetArrayTypeP(arrayDatum);
        n_items = ArrayGetNItems(ARR_NDIM(array), ARR_DIMS(array));
        if(n_items != dimensions) {
            elog(ERROR, "Wrong number of dimensions: %d instead of %d expected", n_items, dimensions);
        }
    }
}

/*
 * Infer the dimensionality of the index from the heap
 */
static int InferDimension(Relation heap, IndexInfo *indexInfo)
{
    int indexCol;

    // If NumIndexAttrs isn't 1 the index has been instantiated on multiple keys and there's no clear way to infer
    // the dim
    if(indexInfo->ii_NumIndexAttrs != 1) {
        return HNSW_DEFAULT_DIM;
    }

    indexCol = indexInfo->ii_IndexAttrNumbers[ 0 ];
    return GetArrayLengthFromHeap(heap, indexCol, indexInfo);
}

/*
 * Initialize the build state
 */
static void InitBuildState(ldb_HnswBuildState *buildstate, Relation heap, Relation index, IndexInfo *indexInfo)
{
    buildstate->heap = heap;
    buildstate->index = index;
    buildstate->indexInfo = indexInfo;
    buildstate->columnType = GetIndexColumnType(index);
    buildstate->dimensions = GetHnswIndexDimensions(index, indexInfo);
    buildstate->index_file_path = ldb_HnswGetIndexFilePath(index);
    buildstate->index_file_fd = -1;
    buildstate->index_buffer_size = 0;
    buildstate->index_buffer = NULL;
    buildstate->external = ldb_HnswGetExternal(index);

    // If a dimension wasn't specified try to infer it
    if(heap != NULL && buildstate->dimensions < 1) {
        buildstate->dimensions = InferDimension(heap, indexInfo);
    }
    /* Require column to have dimensions to be indexed */
    if(buildstate->dimensions < 1) elog(ERROR, "column does not have dimensions, please specify one");

    // not supported because of 8K page limit in postgres WAL pages
    // can pass this limit once quantization is supported
    if(buildstate->dimensions > HNSW_MAX_DIM)
        elog(ERROR,
             "vector dimension %d is too large. "
             "LanternDB currently supports up to %ddim vectors",
             buildstate->dimensions,
             HNSW_MAX_DIM);

    // keeps track of number of tuples added to index
    buildstate->tuples_indexed = 0;
    // number of tuples in the relation to be indexed
    buildstate->reltuples = 0;

    // todo:: add additional possible distance functions to be used in the index
    //  use index_getprocinfo to get the user defined distance functions from relevant index op-class
    //  when added, read the distance function info here once and store it on buildstate for scan to use

    buildstate->tmpCtx
        = AllocSetContextCreate(CurrentMemoryContext, "hnsw build temporary context", ALLOCSET_DEFAULT_SIZES);
}

/*
 * Free resources
 */
static void FreeBuildState(ldb_HnswBuildState *buildstate)
{
    // todo:: in debug/or stats mode collect stats from the tmpCtx before deleting it
    MemoryContextDelete(buildstate->tmpCtx);
}

/*
 * Scan table for tuples to index
 */
static void ScanTable(ldb_HnswBuildState *buildstate)
{
#if PG_VERSION_NUM >= 120000
    buildstate->reltuples = table_index_build_scan(buildstate->heap,
                                                   buildstate->index,
                                                   buildstate->indexInfo,
                                                   true,
                                                   true,
                                                   BuildCallback,
                                                   (void *)buildstate,
                                                   NULL);
#else
    buildstate->reltuples = IndexBuildHeapScan(
        buildstate->heap, buildstate->index, buildstate->indexInfo, true, BuildCallback, (void *)buildstate, NULL);
#endif
}

static void BuildIndexCleanup(ldb_HnswBuildState *buildstate)
{
    usearch_error_t error = NULL;

    if(buildstate->usearch_index) {
        usearch_free(buildstate->usearch_index, &error);
        buildstate->usearch_index = NULL;
    }

    if(buildstate->external_socket && buildstate->external_socket->close) {
        buildstate->external_socket->close(buildstate->external_socket);
    }

    if(buildstate->index_file_fd > 0) {
        // index_file_fd will only exist when we mmap the index file to memory
        if(!buildstate->external && buildstate->index_buffer) {
            int munmap_ret = munmap(buildstate->index_buffer, buildstate->index_buffer_size);
            assert(munmap_ret == 0);
            LDB_UNUSED(munmap_ret);
        }
        close(buildstate->index_file_fd);
    }
}

/*
 * Build the index, writing to the main fork
 */
static void BuildIndex(Relation heap, Relation index, IndexInfo *indexInfo, ldb_HnswBuildState *buildstate)
{
    usearch_error_t        error = NULL;
    usearch_init_options_t opts;
    struct stat            index_file_stat;
    char                  *tmp_index_file_path = NULL;
    const char            *tmp_index_file_fmt_str = "%s/ldb-index-%d.bin";
    // parent_dir + max digits of uint32 (Oid) 10
    const uint32 tmp_index_file_char_cnt = MAXPGPATH + strlen(tmp_index_file_fmt_str) + 10;
    int          munmap_ret;
    metadata_t   metadata;
    uint64       num_added_vectors;

    MemSet(&opts, 0, sizeof(opts));

    InitBuildState(buildstate, heap, index, indexInfo);

    if(buildstate->index_file_path) {
        elog(ERROR,
             "Importing index from file is no longer supported.\n"
             "If you want to use external indexing pass `external=true` in index options");
    }

    opts.dimensions = buildstate->dimensions;

    PopulateUsearchOpts(index, &opts);
    if(opts.pq) {
        buildstate->pq_codebook = load_pq_codebook(index, opts.dimensions, &opts.num_centroids, &opts.num_subvectors);
        assert(0 < opts.num_centroids && opts.num_centroids <= 256);
    }

    buildstate->usearch_scalar = usearch_scalar_f32_k;
    if(opts.metric_kind == usearch_metric_hamming_k) {
        // when using hamming distance, we pass usearch dimension as number of bits
        opts.dimensions *= sizeof(int32) * CHAR_BIT;
        opts.quantization = usearch_scalar_b1_k;
        buildstate->usearch_scalar = usearch_scalar_b1_k;
    }
    // retrievers are not called from here,
    // but we are setting them so the storage layer knows objects are managed
    // externally and does not try to load objects from stream when we call
    // usearch_load
    opts.retriever = ldb_wal_index_node_retriever;
    opts.retriever_mut = ldb_wal_index_node_retriever_mut;

    buildstate->usearch_index = usearch_init(&opts, buildstate->pq_codebook, &error);
    elog(INFO, "done init usearch index");
    assert(error == NULL);

    uint32_t estimated_row_count = EstimateRowCount(heap);

    if(estimated_row_count < EXTERNAL_INDEX_MIN_TUPLES) {
        buildstate->external = false;
    }

    if(buildstate->external) {
        buildstate->external_socket = palloc0(sizeof(external_index_socket_t));
        create_external_index_session(ldb_external_index_host,
                                      ldb_external_index_port,
                                      ldb_external_index_secure,
                                      &opts,
                                      buildstate,
                                      estimated_row_count);
    } else {
        CheckMem(maintenance_work_mem,
                 index,
                 buildstate->usearch_index,
                 estimated_row_count,
                 "index size exceeded maintenance_work_mem during index construction, consider increasing "
                 "maintenance_work_mem");

        usearch_reserve(buildstate->usearch_index, estimated_row_count, &error);

        if(error != NULL) {
            // There's not much we can do if free throws an error, but we want to preserve the contents of the first
            // one in case it does
            usearch_error_t local_error = NULL;
            usearch_free(buildstate->usearch_index, &local_error);
            elog(ERROR, "Error reserving space for index: %s", error);
        }
    }

    UpdateProgress(PROGRESS_CREATEIDX_PHASE, LDB_PROGRESS_HNSW_PHASE_IN_MEMORY_INSERT);
    LanternBench("build hnsw index", ScanTable(buildstate));

    if(!buildstate->external) {
        elog(INFO, "inserted %ld elements", usearch_size(buildstate->usearch_index, &error));
    }

    metadata = usearch_index_metadata(buildstate->usearch_index, &error);
    assert(error == NULL);

    if(buildstate->external) {
        buildstate->index_buffer = palloc(USEARCH_HEADER_SIZE);
        external_index_receive_metadata(
            buildstate->external_socket, &num_added_vectors, &buildstate->index_buffer_size);

        uint32 bytes_read
            = external_index_read_all(buildstate->external_socket, buildstate->index_buffer, USEARCH_HEADER_SIZE);

        if(bytes_read != USEARCH_HEADER_SIZE || LDB_FAILURE_POINT_IS_ENABLED("crash_after_recv_header")) {
            elog(ERROR, "received invalid index header");
        }
    } else {
        // Save index into temporary file
        // To later mmap it into memory
        // The file will be removed in the end
        tmp_index_file_path = palloc0(tmp_index_file_char_cnt);
        // Create index file directory string: $pg_data_dir/ldb_indexes/index-$relfilenode.bin
        snprintf(
            tmp_index_file_path, tmp_index_file_char_cnt, tmp_index_file_fmt_str, DataDir, index->rd_rel->relfilenode);
        usearch_save(buildstate->usearch_index, tmp_index_file_path, &error);
        assert(error == NULL);
        buildstate->index_file_fd = open(tmp_index_file_path, O_RDONLY);
        assert(buildstate->index_file_fd > 0);
    }

    if(!buildstate->external) {
        num_added_vectors = usearch_size(buildstate->usearch_index, &error);
    }

    assert(error == NULL);
    elog(INFO, "done saving %lu vectors", num_added_vectors);

    //****************************** mmap index to memory BEGIN ******************************//
    usearch_free(buildstate->usearch_index, &error);
    assert(error == NULL);
    buildstate->usearch_index = NULL;

    if(!buildstate->external) {
        fstat(buildstate->index_file_fd, &index_file_stat);
        buildstate->index_buffer_size = index_file_stat.st_size;
        buildstate->index_buffer
            = mmap(NULL, index_file_stat.st_size, PROT_READ, MAP_PRIVATE, buildstate->index_file_fd, 0);

        if(buildstate->index_buffer == MAP_FAILED) {
            elog(ERROR, "failed to mmap index file");
        }
    }
    //****************************** mmap index to memory END ******************************//

    // save the index to WAL
    UpdateProgress(PROGRESS_CREATEIDX_PHASE, LDB_PROGRESS_HNSW_PHASE_LOAD);

    if(num_added_vectors == 0) {
        StoreExternalEmptyIndex(index, MAIN_FORKNUM, buildstate->index_buffer, buildstate->dimensions, &opts);
    } else {
        StoreExternalIndex(index,
                           &metadata,
                           MAIN_FORKNUM,
                           buildstate->index_buffer,
                           &opts,
                           buildstate->dimensions,
                           num_added_vectors,
                           buildstate->external_socket,
                           buildstate->index_buffer_size);
    }

    if(!buildstate->external) {
        munmap_ret = munmap(buildstate->index_buffer, buildstate->index_buffer_size);
        assert(munmap_ret == 0);
        LDB_UNUSED(munmap_ret);
        close(buildstate->index_file_fd);
    }

    if(buildstate->external) {
        buildstate->external_socket->close(buildstate->external_socket);
    }

    if(tmp_index_file_path) {
        // remove index file if it was not externally provided
        unlink(tmp_index_file_path);
        pfree(tmp_index_file_path);
    }

    FreeBuildState(buildstate);
}

/*
 * Build an empty index, writing to the init fork
 */
static void BuildEmptyIndex(Relation index, IndexInfo *indexInfo, ldb_HnswBuildState *buildstate)
{
    usearch_error_t        error = NULL;
    usearch_init_options_t opts;
    MemSet(&opts, 0, sizeof(opts));

    InitBuildState(buildstate, NULL, index, indexInfo);
    opts.dimensions = buildstate->dimensions;
    PopulateUsearchOpts(index, &opts);
    // when using hamming distance, we pass dimension as number of bits
    buildstate->usearch_scalar = usearch_scalar_f32_k;
    if(opts.metric_kind == usearch_metric_hamming_k) {
        opts.dimensions *= sizeof(int32) * CHAR_BIT;
        opts.quantization = usearch_scalar_b1_k;
        buildstate->usearch_scalar = usearch_scalar_b1_k;
    }

    if(opts.pq) {
        buildstate->pq_codebook = load_pq_codebook(index, opts.dimensions, &opts.num_centroids, &opts.num_subvectors);
        assert(0 < opts.num_centroids && opts.num_centroids <= 256);
    }

    buildstate->usearch_index = usearch_init(&opts, buildstate->pq_codebook, &error);
    assert(error == NULL);

    char *result_buf = palloc(USEARCH_EMPTY_INDEX_SIZE);
    usearch_save_buffer(buildstate->usearch_index, result_buf, USEARCH_EMPTY_INDEX_SIZE, &error);
    assert(error == NULL && result_buf != NULL);

    StoreExternalEmptyIndex(index, INIT_FORKNUM, result_buf, buildstate->dimensions, &opts);

    usearch_free(buildstate->usearch_index, &error);
    assert(error == NULL);
    buildstate->usearch_index = NULL;

    FreeBuildState(buildstate);
}

/*
 * Build the index for a logged table
 */
IndexBuildResult *ldb_ambuild(Relation heap, Relation index, IndexInfo *indexInfo)
{
    IndexBuildResult  *result;
    ldb_HnswBuildState buildstate;
    memset(&buildstate, 0, sizeof(ldb_HnswBuildState));

    (void)CheckExtensionVersions();
    PG_TRY();
    {
        BuildIndex(heap, index, indexInfo, &buildstate);
    }
    PG_CATCH();
    {
        BuildIndexCleanup(&buildstate);
        PG_RE_THROW();
    }
    PG_END_TRY();
    result = (IndexBuildResult *)palloc(sizeof(IndexBuildResult));
    result->heap_tuples = buildstate.reltuples;
    result->index_tuples = buildstate.tuples_indexed;

    return result;
}

/*
 * Build an empty index for an unlogged table
 */
void ldb_ambuildunlogged(Relation index)
{
    ldb_HnswBuildState buildstate;
    memset(&buildstate, 0, sizeof(ldb_HnswBuildState));
    IndexInfo *indexInfo = BuildIndexInfo(index);
    BuildEmptyIndex(index, indexInfo, &buildstate);
}
