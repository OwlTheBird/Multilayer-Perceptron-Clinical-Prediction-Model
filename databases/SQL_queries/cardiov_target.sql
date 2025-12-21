SELECT SEQN, 
  CASE 
    -- 1. POSITIVE: If ANY column is 1 (Yes), return 1
    WHEN 1 IN (MCQ160B, MCQ160C, MCQ160D, MCQ160E, MCQ160F) THEN 1

    -- 2. NOISE: If NO '1's exist, but we have '7' (Refused), return NULL
    -- We exclude this row because we don't know if the refusal hides a 'Yes'.
    WHEN 7 IN (MCQ160B, MCQ160C, MCQ160D, MCQ160E, MCQ160F) THEN NULL

    -- 3. NEGATIVE: If we are here, we know there are no 1s and no 7s.
    -- If there is ANY data left (which must be 2 or 9), return 0.
    WHEN COALESCE(MCQ160B, MCQ160C, MCQ160D, MCQ160E, MCQ160F) IS NOT NULL THEN 0

    -- 4. EMPTY: If all columns are NULL, return NULL
    ELSE NULL 
  END as has_cardiovascular_disease
FROM HeartQuestions;