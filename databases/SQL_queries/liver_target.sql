-- Liver Target: Extract raw ALT value (LBXSATSI = Alanine Aminotransferase)
-- Classification threshold (40 U/L) will be applied in Python

SELECT 
    SEQN,
    "LBXSATSI (target)" AS ALT_U_L
FROM 
    table_df
WHERE 
    "LBXSATSI (target)" > 0;