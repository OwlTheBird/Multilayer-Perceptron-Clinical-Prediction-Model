SELECT 
    SEQN,
    ln("LBXSATSI") AS ALT_Log

FROM 
    table_df
WHERE 
    "LBXSATSI" > 0;