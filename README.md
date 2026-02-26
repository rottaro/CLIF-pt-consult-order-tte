#PT Consult Order Target Trial Emulation
- Using [MIMIC IV Database](https://physionet.org/content/mimiciv/3.1/)
- Converted to [CLIF Format](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF-MIMIC)
- And dependent on [Eligibility for Mobilization Algorithm](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF-eligibility-for-mobilization/tree/main)

##Required CLIF Tables
### Core pipeline tables

1. **patient**: `patient_id`, `race_category`, `ethnicity_category`, `sex_category`, `death_dttm`
2. **hospitalization**: `patient_id`, `hospitalization_id`, `admission_dttm`, `discharge_dttm`, `discharge_category`, `age_at_admission`
3. **vitals**: `hospitalization_id`, `recorded_dttm`, `vital_category`, `vital_value`
   - `vital_category` = 'heart_rate', 'resp_rate', 'sbp', 'dbp', 'map', 'spo2', 'weight_kg', 'height_cm'
4. **labs**: `hospitalization_id`, `lab_result_dttm`, `lab_order_dttm`, `lab_category`, `lab_value`, `lab_value_numeric`
   - `lab_category` = 'lactate', 'creatinine', 'bilirubin_total', 'po2_arterial', 'platelet_count'
5. **medication_admin_continuous**: `hospitalization_id`, `admin_dttm`, `med_name`, `med_category`, `med_dose`, `med_dose_unit`, `med_group`
   - `med_category` = "norepinephrine", "epinephrine", "phenylephrine", "vasopressin", "dopamine", "angiotensin", "nicardipine", "nitroprusside", "clevidipine", "cisatracurium", "vecuronium", "rocuronium"
6. **respiratory_support**: `hospitalization_id`, `recorded_dttm`, `device_category`, `mode_category`, `tracheostomy`, `fio2_set`, `lpm_set`, `resp_rate_set`, `peep_set`, `resp_rate_obs`
7. **crrt_therapy**: `hospitalization_id`, `recorded_dttm`
8. **key_icu_orders**: `hospitalization_id`,'order_dttm', 'order_category'# CLIF-pt-consult-order-tte
