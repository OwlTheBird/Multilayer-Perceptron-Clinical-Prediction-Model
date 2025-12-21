SELECT 
    SEQN,
    ln("LBXSATSI") AS liver_alt_U_L

FROM 
    table_df
WHERE 
    "LBXSATSI" > 0;