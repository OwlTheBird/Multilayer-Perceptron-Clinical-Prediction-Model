SELECT *
FROM Demographics AS Demo

LEFT JOIN Vitals AS Vit
    ON Demo.SEQN = Vit.SEQN

LEFT JOIN Albumin_Creatinie AS AC
    ON Demo.SEQN = AC.SEQN

LEFT JOIN [Complete Blood Count] AS CBC
    ON Demo.SEQN = CBC.SEQN

LEFT JOIN [Total Cholesterol] AS TC
    ON Demo.SEQN = TC.SEQN

LEFT JOIN AlcholUsage AS AU
    ON Demo.SEQN = AU.SEQN

LEFT JOIN [Biochem Profile] AS BP
    ON Demo.SEQN = BP.SEQN

LEFT JOIN HDL_Cholesterol AS HDL
    ON Demo.SEQN = HDL.SEQN

LEFT JOIN [Body Measures] AS BM
    ON Demo.SEQN = BM.SEQN

LEFT JOIN Smoke AS S
    ON Demo.SEQN = S.SEQN

LEFT JOIN HeartQuestions AS HQs
    ON Demo.SEQN = HQs.SEQN;
--limit 10; 