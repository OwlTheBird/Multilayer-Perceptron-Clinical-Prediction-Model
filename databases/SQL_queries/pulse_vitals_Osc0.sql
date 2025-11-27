SELECT ("BPXSY1 (target)" + "BPXSY2 (target)" + "BPXSY3 (target)") / 3.0 AS row_avg
FROM Vitals
WHERE Is_Oscillometric = 0;

-- note if one of the selected targets is null it will just add it to the avg for example null + null + 3 / 3 = 3
-- for now i will just ignore it and fix it later the query later or treat it in python code