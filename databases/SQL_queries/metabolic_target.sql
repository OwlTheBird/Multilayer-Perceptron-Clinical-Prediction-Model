SELECT
    -- 1. Waist Label (Abdominal Obesity)
    -- NCEP ATP III: > 102cm (Men), > 88cm (Women)
    CASE
        WHEN RIAGENDR = 1 AND "BMXWAIST (target)" > 102 THEN 1
        WHEN RIAGENDR = 2 AND "BMXWAIST (target)" > 88  THEN 1
        WHEN Waist IS NULL OR RIAGENDR IS NULL THEN NULL
        ELSE 0
    END AS Waist_Label,

    -- 2. Triglycerides Label
    -- NCEP ATP III: >= 150 mg/dL
    CASE
        WHEN "LBXTLG (target)" >= 150 THEN 1
        WHEN "LBXTLG (target)" IS NULL THEN NULL
        ELSE 0
    END AS Triglycerides_Label,

    -- 3. HDL Label (Reduced HDL)
    -- NCEP ATP III: < 40 mg/dL (Men), < 50 mg/dL (Women)
    CASE
        WHEN RIAGENDR = 1 AND "LBDHDD (target)" < 40 THEN 1
        WHEN RIAGENDR = 2 AND "LBDHDD (target)" < 50 THEN 1
        WHEN "LBDHDD (target)" IS NULL OR RIAGENDR IS NULL THEN NULL
        ELSE 0
    END AS HDL_Label,

    -- 4. Blood Pressure Label
    -- NCEP ATP III: >= 130 mmHg Systolic OR >= 85 mmHg Diastolic
    CASE
        -- If EITHER is high, the condition is met (Result: 1)
        WHEN Final_Harmonized_Systolic >= 130 OR Final_Harmonized_Diastolic >= 85 THEN 1
        
        WHEN Final_Harmonized_Systolic IS NULL OR Final_Harmonized_Diastolic IS NULL THEN NULL
        
        -- Only return 0 if both values exist and both are normal
        ELSE 0
    END AS BP_Label,

    -- 5. Glucose Label (Elevated Fasting Glucose)
    -- NCEP ATP III (2005 Revision): >= 100 mg/dL
    CASE
        WHEN Fasting_Glucose >= 100 THEN 1
        WHEN Fasting_Glucose IS NULL THEN NULL
        ELSE 0
    END AS Glucose_Label

FROM table_df;