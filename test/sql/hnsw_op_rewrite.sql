---------------------------------------------------------------------
-- Test Database Crashes which were caused by operator rewriting logic
---------------------------------------------------------------------
-- This case were causing Segfault from
-- post_parse_analyze_hook_with_operator_check() -> ldb_get_operator_oids() -> ... LookupOperName() ... -> GetRealCmin()
BEGIN;
DROP EXTENSION IF EXISTS lantern CASCADE;
CREATE EXTENSION lantern;
\set ON_ERROR_STOP off
SELECT ARRAY[1,1] <-> ARRAY[1,1];
ROLLBACK;

-- This case were causing: ERROR:  unrecognized node type: 233
-- And sometimes Segfault as well
-- This is caused when trying to call expression_tree_mutator with OidList_T node
BEGIN;
\set ON_ERROR_STOP off
DROP TABLE IF EXISTS t1 CASCADE;
CREATE TABLE t1 (
    id TEXT PRIMARY KEY,
    v REAL[]
);

DROP TABLE IF EXISTS t2 CASCADE;
CREATE TABLE t2 (
    id SERIAL PRIMARY KEY,
    t1_id TEXT,
    CONSTRAINT fk_t1 FOREIGN KEY(t1_id) REFERENCES t1(id)
);

INSERT INTO t1 (id, v) VALUES ('1', ARRAY[0,0,0]);

CREATE INDEX ON t1 USING hnsw(v dist_cos_ops) WITH (m=32, ef_construction=128, ef=64);

INSERT INTO t2 (t1_id) VALUES ('1');
INSERT INTO t2 (t1_id) VALUES ('1');
INSERT INTO t2 (t1_id) VALUES ('1');
INSERT INTO t2 (t1_id) VALUES ('1');
INSERT INTO t2 (t1_id) VALUES ('1');
INSERT INTO t2 (t1_id) VALUES ('1');
INSERT INTO t2 (t1_id) VALUES ('1');
INSERT INTO t2 (t1_id) VALUES ('1');
INSERT INTO t2 (t1_id) VALUES ('1');
INSERT INTO t2 (t1_id) VALUES ('1');
END;

