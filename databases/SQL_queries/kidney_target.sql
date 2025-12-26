-- Kidney Target: Calculate ACR (Albumin-to-Creatinine Ratio)
-- URXUMA: Albumin, urine (ug/mL) | URXUCR: Creatinine, urine (mg/dL)
-- Conversion: ACR (mg/g) = (URXUMA / URXUCR) * 100

SELECT
    SEQN, 
    CASE
        WHEN "URXUMA (target)" IS NOT NULL AND "URXUCR (target)" IS NOT NULL
        THEN (("URXUMA (target)" / "URXUCR (target)") * 100.00)
        ELSE NULL
    END AS ACR_mg_g
FROM table_df
WHERE "URXUCR (target)" > 0;
