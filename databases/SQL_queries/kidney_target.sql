-- so we must divide URXUMA/10 we did it by multiplying URXUCR*10
    -- to convert mg to ug we need *1000 and from dl to mL we need /100
    --so in total we have * by 10 but for it to become per gram we need in total to multiply *100
SELECT --note that URXUMA: Albumin, urine (ug/mL) | URXUCR - Creatinine, urine (mg/dL)
    SEQN, 
    CASE
        WHEN 
            "URXUMA (target)" IS NOT NULL AND "URXUCR (target)" IS NOT NULL
        THEN 
            (("URXUMA (target)" / "URXUCR (target)") * 100.00)
        ELSE 
            NULL
        END AS ACR_mg_g,
    CASE
        WHEN
            "URXUMA (target)" IS NOT NULL AND "URXUCR (target)" IS NOT NULL
        THEN
            ln( 1 + ("URXUMA (target)" / "URXUCR (target)") * 100.00 )
        ELSE
            NULL
        END AS ACR_Log
FROM table_df
WHERE "URXUCR (target)" > 0;
