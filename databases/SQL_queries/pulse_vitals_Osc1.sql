SELECT ("BPXSY1 (target)" + "BPXSY2 (target)" + "BPXSY3 (target)") / 3.0 AS row_avg
FROM Vitals
WHERE Is_Oscillometric = 1;