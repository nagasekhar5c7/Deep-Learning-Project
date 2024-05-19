import os
import os.path as ops
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

mimic_base_path = r'project/mimiciv1.0'
inputs_base_path = r'project/inputs'
features_base_path = r'preprocess1.0/Processed_files'

PROC_FILE = os.path.join(mimic_base_path, 'procedures_icd.csv.gz')  # ICD codes per
MED_FILE = os.path.join(mimic_base_path, 'prescriptions.csv.gz')  # NDC codes per admission/patient
DIAG_FILE = os.path.join(mimic_base_path, 'diagnoses_icd.csv.gz')  # ICD codes per admission/patient
LAB_EVENTS = os.path.join(mimic_base_path, 'labevents.csv.gz')
PATIENT_FILE = os.path.join(mimic_base_path, 'patients.csv.gz')
ICU_FILE = os.path.join(mimic_base_path, 'icustays.csv.gz')
CHART_EVENTS = os.path.join(mimic_base_path, 'chartevents.csv.gz')
ADM_FILE = os.path.join(mimic_base_path, 'admissions.csv.gz')

# From step 1, call combine_codes which will give a dataset with all event codes, for every patient and admission
# from Preprocess_Step_1 import combine_codes
# combined = combine_codes()

patient_codes_cleaned_path = os.path.join(features_base_path, 'patient_codes_cleaned_step1.pickle')
with open(patient_codes_cleaned_path, 'rb') as file:
    combined = pickle.load(file)

# Categories of CVD diseases along with their ICD codes

hyp = ['401', '4010', '4011', '4019',
       '402', '4020', '40200', '40201', '4021', '40210', '40211', '4029', '40290', '40291',
       '403', '4030', '40300', '40301', '4031', '40310', '40311', '4039', '40390', '40391',
       '404', '4040', '40400', '40401', '40402', '40403', '4041', '40410', '40411', '40412', '40413', '4049', '40490', '40491', '40492', '40493',
       '405', '4050', '40501', '40509', '4051', '40511', '40519', '4059', '40591', '40599',
       'I10', 'I11', 'I110', 'I119',
       'I12', 'I120', 'I129', 'I13', 'I130', 'I131', 'I1310', 'I1311', 'I132',
       'I15', 'I150', 'I151', 'I152', 'I158', 'I159', 'I16', 'I160', 'I161', 'I169']

con = ['39891', 'I098', 'I0981', 'I0989',
       '428', '4280', '4281', '4282', '42820', '42821', '42822', '42823', '4283', '42830', '42831', '42832', '42833',
       '4284', '42840', '42841', '42842', '42843', '4289',
       'I50', 'I501', 'I502', 'I5020', 'I5021',
       'I5022', 'I5023', 'I503', 'I5030', 'I5031', 'I5032', 'I5033', 'I504', 'I5040', 'I5041', 'I5042', 'I5043', 'I508',
       'I5081', 'I50810', 'I50811', 'I50812',
       'I50813', 'I50814', 'I5082', 'I5083', 'I5084', 'I5089', 'I509']

afib = ['I48', 'I480', 'I481', 'I4811', 'I4819', 'I4820', 'I4821', 'I489', 'I4891', '42731']

# cor_art = ['41400', '41401', '41402', '41403', '41404', '41405', '41406', '41407', ' 4143', '4144', ' I2583', ' I2584',
#            'I251', ' I2510', 'I2511', 'I25110', 'I25111', 'I25118', 'I25119']

cor_art = ['414', '4140', '41400', '41401', '41402', '41403', '41404', '41405', '41406', '41407', '4141',
           '41410', '41411', '41412', '41419', '4142', '4143', '4144', '4148', '4149',
           'I25', 'I251', 'I2510', 'I2511', 'I25110', 'I25111', 'I25112', 'I25118', 'I25119', 'I252', 'I253', 'I254', 'I2541',
           'I2542', 'I255', 'I256', 'I257', 'I2570', 'I25700', 'I25701', 'I25702', 'I25708', 'I25709', 'I2571', 'I25710',
           'I25711', 'I25712', 'I25718', 'I25719', 'I2572', 'I25720', 'I25721', 'I25722', 'I25728', 'I25729', 'I2573', 'I25730',
           'I25731', 'I25732', 'I25738', 'I25739', 'I2575', 'I25750', 'I25751', 'I25752', 'I25758', 'I25759', 'I2576', 'I25760',
           'I25761', 'I25762', 'I25768', 'I25769', 'I2579', 'I25790', 'I25791', 'I25792', 'I25798', 'I25799', 'I258', 'I2581',
           'I25810', 'I25811', 'I25812', 'I2582', 'I2583', 'I2584', 'I2589', 'I259']
hyp_set = set(hyp)
con_set = set(con)
afib_set = set(afib)
cor_art_set = set(cor_art)


# Method to read all chart events from chartevents table
def get_combined_clinical_df():
    # Big dataset so read it in chunks
    # Filtering out only the chart events we need
    clin_events_pd = pd.DataFrame()
    i = 0
    for chunk in pd.read_csv(CHART_EVENTS, chunksize=6000000, low_memory=False):
        chunk = chunk[(chunk['subject_id'].notnull()) & (chunk['hadm_id'].notnull())]
        chunk = chunk[
            (chunk['itemid'] == 221) | (chunk['itemid'] == 220045) | (chunk['itemid'] == 51) | (chunk['itemid'] == 442)
            | (chunk['itemid'] == 455) | (chunk['itemid'] == 6701) | (chunk['itemid'] == 220179) | (
                        chunk['itemid'] == 220050)
            | (chunk['itemid'] == 8368) | (chunk['itemid'] == 8440) | (chunk['itemid'] == 8555) | (
                        chunk['itemid'] == 220180)
            | (chunk['itemid'] == 220051) | (chunk['itemid'] == 223761) | (chunk['itemid'] == 678) | (
                        chunk['itemid'] == 223762)
            | (chunk['itemid'] == 676) | (chunk['itemid'] == 615) | (chunk['itemid'] == 618)
            | (chunk['itemid'] == 220210) | (chunk['itemid'] == 224690) | (chunk['itemid'] == 807) | (
                        chunk['itemid'] == 811)
            | (chunk['itemid'] == 1529) | (chunk['itemid'] == 3745) | (chunk['itemid'] == 3744) | (
                        chunk['itemid'] == 225664)
            | (chunk['itemid'] == 220621) | (chunk['itemid'] == 226537)]
        print(i)
        i = i + 1
        clin_events_pd = pd.concat([clin_events_pd, chunk])

    clin_events_pd = clin_events_pd.drop(columns=['stay_id', 'charttime', 'storetime', 'value', 'valueuom', 'warning'])
    clin_events_pd = clin_events_pd.reset_index()

    # grouping of Heart Rate for every patient and their admission
    grouped_multiple_pd_hr = clin_events_pd[
        (clin_events_pd['itemid'] == 211) | (clin_events_pd['itemid'] == 220045)].groupby(
        ['subject_id', 'hadm_id']).agg({'valuenum': ['mean', 'min', 'max']})
    grouped_multiple_pd_hr.columns = ['HR_mean', 'HR_min', 'HR_max']
    grouped_multiple_pd_hr = grouped_multiple_pd_hr.reset_index()
    grouped_multiple_pd_hr.columns = grouped_multiple_pd_hr.columns.str.upper()

    # grouping of Sys BP for every patient and their admission
    grouped_multiple_pd_SysBP = clin_events_pd[
        clin_events_pd['itemid'].isin([51, 442, 455, 6701, 220179, 220050])].groupby(['subject_id', 'hadm_id']).agg(
        {'valuenum': ['mean', 'min', 'max']})
    grouped_multiple_pd_SysBP.columns = ['SysBP_mean', 'SysBP_min', 'SysBP_max']
    grouped_multiple_pd_SysBP = grouped_multiple_pd_SysBP.reset_index()
    grouped_multiple_pd_SysBP.columns = grouped_multiple_pd_SysBP.columns.str.upper()

    # grouping of Dias BP for every patient and their admission
    grouped_multiple_pd_DiasBP = clin_events_pd[
        clin_events_pd['itemid'].isin([8368, 8440, 8441, 8555, 220180, 220051])].groupby(['subject_id', 'hadm_id']).agg(
        {'valuenum': ['mean', 'min', 'max']})
    grouped_multiple_pd_DiasBP.columns = ['DiasBP_mean', 'DiasBP_min', 'DiasBP_max']
    grouped_multiple_pd_DiasBP = grouped_multiple_pd_DiasBP.reset_index()
    grouped_multiple_pd_DiasBP.columns = grouped_multiple_pd_DiasBP.columns.str.upper()

    # grouping of Respiratory rate for every patient and their admission
    grouped_multiple_pd_RespRate = clin_events_pd[clin_events_pd['itemid'].isin([615, 618, 220210, 224690])].groupby(
        ['subject_id', 'hadm_id']).agg({'valuenum': ['mean', 'min', 'max']})
    grouped_multiple_pd_RespRate.columns = ['RespRate_mean', 'RespRate_min', 'RespRate_max']
    grouped_multiple_pd_RespRate = grouped_multiple_pd_RespRate.reset_index()
    grouped_multiple_pd_RespRate.columns = grouped_multiple_pd_RespRate.columns.str.upper()

    # grouping of Glucose for every patient and their admission
    grouped_multiple_pd_Glucose = clin_events_pd[
        clin_events_pd['itemid'].isin([807, 811, 1529, 3745, 3744, 225664, 220621, 226537])].groupby(
        ['subject_id', 'hadm_id']).agg({'valuenum': ['mean', 'min', 'max']})
    grouped_multiple_pd_Glucose.columns = ['Glucose_mean', 'Glucose_min', 'Glucose_max']
    grouped_multiple_pd_Glucose = grouped_multiple_pd_Glucose.reset_index()
    grouped_multiple_pd_Glucose.columns = grouped_multiple_pd_Glucose.columns.str.upper()

    # grouping of body temperature for every patient and their admission
    temperature_pd = clin_events_pd[clin_events_pd['itemid'].isin([223761, 678, 223762, 676])]
    temperature_pd['valuenum'] = temperature_pd['valuenum'].apply(
        lambda x: (x - 32.00) / 1.8 if (x > 70.00 and x < 120.00) else x)
    grouped_multiple_pd_Temp = temperature_pd.groupby(['subject_id', 'hadm_id']).agg(
        {'valuenum': ['mean', 'min', 'max']})
    grouped_multiple_pd_Temp.columns = ['Temp_mean', 'Temp_min', 'Temp_max']
    grouped_multiple_pd_Temp = grouped_multiple_pd_Temp.reset_index()
    grouped_multiple_pd_Temp.columns = grouped_multiple_pd_Temp.columns.str.upper()

    # Merge all the tables with combined table from preprocess step 1, on Subject Id and Admission Id with a left join
    final_merged_clin_events = combined.merge(grouped_multiple_pd_hr, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    final_merged_clin_events = final_merged_clin_events.merge(grouped_multiple_pd_SysBP, on=['SUBJECT_ID', 'HADM_ID'],
                                                              how='left')
    final_merged_clin_events = final_merged_clin_events.merge(grouped_multiple_pd_DiasBP, on=['SUBJECT_ID', 'HADM_ID'],
                                                              how='left')
    final_merged_clin_events = final_merged_clin_events.merge(grouped_multiple_pd_RespRate,
                                                              on=['SUBJECT_ID', 'HADM_ID'], how='left')
    final_merged_clin_events = final_merged_clin_events.merge(grouped_multiple_pd_Glucose, on=['SUBJECT_ID', 'HADM_ID'],
                                                              how='left')
    final_merged_clin_events = final_merged_clin_events.merge(grouped_multiple_pd_Temp, on=['SUBJECT_ID', 'HADM_ID'],
                                                              how='left')

    return final_merged_clin_events


# Method to read all lab events from labevents table
def get_combined_lab_df():
    lab_events_pd = pd.DataFrame()
    i = 0
    # big dataset so reading it chunkwise
    for chunk in pd.read_csv(LAB_EVENTS, chunksize=6000000, low_memory=False):
        chunk = chunk[(chunk['subject_id'].notnull()) & (chunk['hadm_id'].notnull())]

        # filtering out only for required lab values
        chunk = chunk[(chunk['itemid'] == 51265) | (chunk['itemid'] == 50960) | (chunk['itemid'] == 50862) | (
                    chunk['itemid'] == 50893)
                      | (chunk['itemid'] == 51006) | (chunk['itemid'] == 50889) | (chunk['itemid'] == 50904) | (
                                  chunk['itemid'] == 50906)
                      | (chunk['itemid'] == 50907) | (chunk['itemid'] == 51000) | (chunk['itemid'] == 50963)]

        # removing irrelevant columns since valuenum gives the value we need
        chunk = chunk.drop(columns=['labevent_id', 'specimen_id', 'storetime', 'value', 'valueuom', 'charttime',
                                    'ref_range_lower', 'ref_range_upper', 'flag', 'priority', 'comments'])
        print(i)
        lab_events_pd = pd.concat([lab_events_pd, chunk])
        i = i + 1
        # if(i > 0):
    # break

    lab_events_pd = lab_events_pd.reset_index()
    lab_events_pd.hadm_id = lab_events_pd.hadm_id.astype(int)

    # 51265 Item ID is for Platelets
    grouped_multiple_pd_platelets = lab_events_pd[lab_events_pd['itemid'] == 51265].groupby(
        ['subject_id', 'hadm_id']).agg({'valuenum': ['mean', 'min', 'max']})
    grouped_multiple_pd_platelets.columns = ['platelets_mean', 'platelets_min', 'platelets_max']
    grouped_multiple_pd_platelets = grouped_multiple_pd_platelets.reset_index()
    grouped_multiple_pd_platelets.columns = grouped_multiple_pd_platelets.columns.str.upper()

    # 50960 Item ID is for Magnesium
    grouped_multiple_pd_magnesium = lab_events_pd[lab_events_pd['itemid'] == 50960].groupby(
        ['subject_id', 'hadm_id']).agg({'valuenum': ['mean', 'min', 'max']})
    grouped_multiple_pd_magnesium.columns = ['magnesium_mean', 'magnesium_min', 'magnesium_max']
    grouped_multiple_pd_magnesium = grouped_multiple_pd_magnesium.reset_index()
    grouped_multiple_pd_magnesium.columns = grouped_multiple_pd_magnesium.columns.str.upper()

    # 50862 Item ID is for Albumin
    grouped_multiple_pd_albumin = lab_events_pd[lab_events_pd['itemid'] == 50862].groupby(
        ['subject_id', 'hadm_id']).agg({'valuenum': ['mean', 'min', 'max']})
    grouped_multiple_pd_albumin.columns = ['albumin_mean', 'albumin_min', 'albumin_max']
    grouped_multiple_pd_albumin = grouped_multiple_pd_albumin.reset_index()
    grouped_multiple_pd_albumin.columns = grouped_multiple_pd_albumin.columns.str.upper()

    # 50893 Item ID is for Calcium
    grouped_multiple_pd_calcium = lab_events_pd[lab_events_pd['itemid'] == 50893].groupby(
        ['subject_id', 'hadm_id']).agg({'valuenum': ['mean', 'min', 'max']})
    grouped_multiple_pd_calcium.columns = ['calcium_mean', 'calcium_min', 'calcium_max']
    grouped_multiple_pd_calcium = grouped_multiple_pd_calcium.reset_index()
    grouped_multiple_pd_calcium.columns = grouped_multiple_pd_calcium.columns.str.upper()

    # 51006 Item ID is for Urea
    grouped_multiple_pd_urea_N = lab_events_pd[lab_events_pd['itemid'] == 51006].groupby(['subject_id', 'hadm_id']).agg(
        {'valuenum': ['mean', 'min', 'max']})
    grouped_multiple_pd_urea_N.columns = ['urea_N_mean', 'urea_N_min', 'urea_N_max']
    grouped_multiple_pd_urea_N = grouped_multiple_pd_urea_N.reset_index()
    grouped_multiple_pd_urea_N.columns = grouped_multiple_pd_urea_N.columns.str.upper()

    # 50889 Item ID is for C-Reactive Protein
    grouped_multiple_pd_crp = lab_events_pd[lab_events_pd['itemid'] == 50889].groupby(['subject_id', 'hadm_id']).agg(
        {'valuenum': ['mean', 'min', 'max']})
    grouped_multiple_pd_crp.columns = ['crp_mean', 'crp_min', 'crp_max']
    grouped_multiple_pd_crp = grouped_multiple_pd_crp.reset_index()
    grouped_multiple_pd_crp.columns = grouped_multiple_pd_crp.columns.str.upper()

    # 50904 Item ID is for Cholesterol, HDL
    grouped_multiple_pd_hdl = lab_events_pd[lab_events_pd['itemid'] == 50904].groupby(['subject_id', 'hadm_id']).agg(
        {'valuenum': ['mean', 'min', 'max']})
    grouped_multiple_pd_hdl.columns = ['hdl_mean', 'hdl_min', 'hdl_max']
    grouped_multiple_pd_hdl = grouped_multiple_pd_hdl.reset_index()
    grouped_multiple_pd_hdl.columns = grouped_multiple_pd_hdl.columns.str.upper()

    # 50906 Item ID is for Cholesterol, LDL
    grouped_multiple_pd_ldl = lab_events_pd[lab_events_pd['itemid'] == 50906].groupby(['subject_id', 'hadm_id']).agg(
        {'valuenum': ['mean', 'min', 'max']})
    grouped_multiple_pd_ldl.columns = ['ldl_mean', 'ldl_min', 'ldl_max']
    grouped_multiple_pd_ldl = grouped_multiple_pd_ldl.reset_index()
    grouped_multiple_pd_ldl.columns = grouped_multiple_pd_ldl.columns.str.upper()

    # 50907 Item ID is for Cholesterol, Total
    grouped_multiple_pd_totalchl = lab_events_pd[lab_events_pd['itemid'] == 50907].groupby(
        ['subject_id', 'hadm_id']).agg({'valuenum': ['mean', 'min', 'max']})
    grouped_multiple_pd_totalchl.columns = ['totalchl_mean', 'totalchl_min', 'totalchl_max']
    grouped_multiple_pd_totalchl = grouped_multiple_pd_totalchl.reset_index()
    grouped_multiple_pd_totalchl.columns = grouped_multiple_pd_totalchl.columns.str.upper()

    # 51000 Item ID is for Triglycerides
    grouped_multiple_pd_trigly = lab_events_pd[lab_events_pd['itemid'] == 51000].groupby(['subject_id', 'hadm_id']).agg(
        {'valuenum': ['mean', 'min', 'max']})
    grouped_multiple_pd_trigly.columns = ['trigly_mean', 'trigly_min', 'trigly_max']
    grouped_multiple_pd_trigly = grouped_multiple_pd_trigly.reset_index()
    grouped_multiple_pd_trigly.columns = grouped_multiple_pd_trigly.columns.str.upper()

    # 50963 Item ID is for NTproBNP
    grouped_multiple_pd_ntprobnp = lab_events_pd[lab_events_pd['itemid'] == 50963].groupby(
        ['subject_id', 'hadm_id']).agg({'valuenum': ['mean', 'min', 'max']})
    grouped_multiple_pd_ntprobnp.columns = ['ntprobnp_mean', 'ntprobnp_min', 'ntprobnp_max']
    grouped_multiple_pd_ntprobnp = grouped_multiple_pd_ntprobnp.reset_index()
    grouped_multiple_pd_ntprobnp.columns = grouped_multiple_pd_ntprobnp.columns.str.upper()

    chart_df = get_combined_clinical_df()

    # Merge all the tables with combined table from preprocess step 1, on Subject Id and Admission Id with a left join
    final_merged_lab_events = chart_df.merge(grouped_multiple_pd_platelets, on=['SUBJECT_ID', 'HADM_ID'], how='left')
    final_merged_lab_events = final_merged_lab_events.merge(grouped_multiple_pd_magnesium, on=['SUBJECT_ID', 'HADM_ID'],
                                                            how='left')
    final_merged_lab_events = final_merged_lab_events.merge(grouped_multiple_pd_albumin, on=['SUBJECT_ID', 'HADM_ID'],
                                                            how='left')
    final_merged_lab_events = final_merged_lab_events.merge(grouped_multiple_pd_calcium, on=['SUBJECT_ID', 'HADM_ID'],
                                                            how='left')
    final_merged_lab_events = final_merged_lab_events.merge(grouped_multiple_pd_urea_N, on=['SUBJECT_ID', 'HADM_ID'],
                                                            how='left')
    final_merged_lab_events = final_merged_lab_events.merge(grouped_multiple_pd_crp, on=['SUBJECT_ID', 'HADM_ID'],
                                                            how='left')
    final_merged_lab_events = final_merged_lab_events.merge(grouped_multiple_pd_hdl, on=['SUBJECT_ID', 'HADM_ID'],
                                                            how='left')
    final_merged_lab_events = final_merged_lab_events.merge(grouped_multiple_pd_ldl, on=['SUBJECT_ID', 'HADM_ID'],
                                                            how='left')
    final_merged_lab_events = final_merged_lab_events.merge(grouped_multiple_pd_totalchl, on=['SUBJECT_ID', 'HADM_ID'],
                                                            how='left')
    final_merged_lab_events = final_merged_lab_events.merge(grouped_multiple_pd_trigly, on=['SUBJECT_ID', 'HADM_ID'],
                                                            how='left')
    final_merged_lab_events = final_merged_lab_events.merge(grouped_multiple_pd_ntprobnp, on=['SUBJECT_ID', 'HADM_ID'],
                                                            how='left')

    return final_merged_lab_events


# Get all the admissions properly for each patient having more than one admissions
def get_chart_lab_combined():
    chart_lab_combined_df = get_combined_lab_df()

    # print(chart_df.info())
    # print(lab_df.info())

    # merge the two on subject id and admission id with a left join
    # chart_lab_combined_df = chart_df.merge(lab_df, on = ['SUBJECT_ID', 'HADM_ID'], how = 'left')
    # print(chart_lab_combined_df.info())

    # Filter for all subject ids where more than one admission exists
    chart_lab_combined_df_multi_admits = chart_lab_combined_df.groupby("SUBJECT_ID").filter(lambda x: len(x) > 1)

    # Compare with admissions table for all the other subject ids along with count of their admissions, so that we only take all the subjects that have all the admissions present from the admission table and not just subset of them.
    admissions_df = pd.read_csv(ADM_FILE)
    admissions_df = admissions_df.drop(
        columns=['admission_location', 'discharge_location', 'language', 'edregtime', 'edouttime', 'deathtime'])
    admissions_df.columns = admissions_df.columns.str.upper()
    admissions_count = admissions_df.groupby('SUBJECT_ID').size().reset_index(name='counts')

    final_merged_multi_visits_count = chart_lab_combined_df_multi_admits.groupby('SUBJECT_ID').size().reset_index(
        name='counts')
    merged_counts_df = final_merged_multi_visits_count.merge(admissions_count, on=["SUBJECT_ID", "counts"], how="inner")
    final_merged_with_admissions = chart_lab_combined_df_multi_admits[
        chart_lab_combined_df_multi_admits['SUBJECT_ID'].isin(merged_counts_df.SUBJECT_ID.unique())]
    final_merged_with_admissions = final_merged_with_admissions.merge(admissions_df, on=['SUBJECT_ID', 'HADM_ID'],
                                                                      how='left')

    # Return combined df with all the chart events, lab events, event codes and with proper number of admissions for each patient
    return final_merged_with_admissions


# Method to assign labels to all the admissions. Label Future Readmit given as if next admission was under 30 days, and if the patient was readmitted with a similar category of CVD diseases, then Future Readmit = 1, else = 0
def assign_Labels():
    final_merged_with_admissions = get_chart_lab_combined()
    final_merged_with_admissions.info()
    # convert admit and disch time to pd datetime format
    final_merged_with_admissions.ADMITTIME = pd.to_datetime(final_merged_with_admissions.ADMITTIME,
                                                            format='%Y-%m-%d %H:%M:%S', errors='coerce')
    final_merged_with_admissions.DISCHTIME = pd.to_datetime(final_merged_with_admissions.DISCHTIME,
                                                            format='%Y-%m-%d %H:%M:%S', errors='coerce')
    final_merged_with_admissions = final_merged_with_admissions.reset_index()

    # loop through all the rows in the dataframe and check with the next admission of same subject
    for idx, row in final_merged_with_admissions.iterrows():
        final_merged_with_admissions.loc[idx, 'FUTURE_READMIT'] = 0
        if idx < final_merged_with_admissions.shape[0] - 1:

            if final_merged_with_admissions.loc[idx, 'SUBJECT_ID'] == final_merged_with_admissions.loc[
             idx + 1, 'SUBJECT_ID']:
                inter_check = False
                currentDisch = final_merged_with_admissions.loc[idx, 'DISCHTIME'].to_pydatetime().date()
                nextAdmitTime = final_merged_with_admissions.loc[idx + 1, 'ADMITTIME'].to_pydatetime().date()
                delta = nextAdmitTime - currentDisch
                Current_ICD_Codes_List = final_merged_with_admissions.loc[idx, 'CODE_ICD']
                Current_ICD_Codes_List = [s.replace("9.", "") for s in Current_ICD_Codes_List]
                Current_ICD_Codes_List = [s.replace("10.", "") for s in Current_ICD_Codes_List]

                Next_ICD_Codes_List = final_merged_with_admissions.loc[idx + 1, 'CODE_ICD']
                Next_ICD_Codes_List = [s.replace("9.", "") for s in Next_ICD_Codes_List]
                Next_ICD_Codes_List = [s.replace("10.", "") for s in Next_ICD_Codes_List]

                Current_ICD_Codes_Set = set(Current_ICD_Codes_List)
                Next_ICD_Codes_Set = set(Next_ICD_Codes_List)

                # Comparison with category of disease
                if ((len(Current_ICD_Codes_Set.intersection(hyp_set)) > 0 and len(
                        Next_ICD_Codes_Set.intersection(hyp_set)) > 0) or
                        (len(Current_ICD_Codes_Set.intersection(con_set)) > 0 and len(
                            Next_ICD_Codes_Set.intersection(con_set)) > 0) or
                        (len(Current_ICD_Codes_Set.intersection(afib_set)) > 0 and len(
                            Next_ICD_Codes_Set.intersection(afib_set)) > 0) or
                        (len(Current_ICD_Codes_Set.intersection(cor_art_set)) > 0 and len(
                            Next_ICD_Codes_Set.intersection(cor_art_set)) > 0)):
                    inter_check = True

                # Setting label as 1
                if inter_check == True and delta.days <= 30:
                    final_merged_with_admissions.loc[idx, 'FUTURE_READMIT'] = 1

    # Used Xgboost to find out the most important features, and selected the top 20 lab/chart events and the remaining are the event codes and other features
    final_merged_with_admissions_labels_sel_features = final_merged_with_admissions[
        ['SUBJECT_ID', 'GENDER','HADM_ID', 'PRO_CODE', 'CODE_ICD', 'NDC', 'DOSES_PER_24_HRS', 'ADMITTIME', 'DISCHTIME','ANCHOR_AGE',
         'MAGNESIUM_MEAN', 'MAGNESIUM_MAX',
         'PLATELETS_MEAN', 'PLATELETS_MIN', 'PLATELETS_MAX', 'CALCIUM_MEAN',
         'CALCIUM_MIN', 'CALCIUM_MAX', 'UREA_N_MEAN', 'UREA_N_MIN', 'UREA_N_MAX',
         'ALBUMIN_MEAN', 'ALBUMIN_MAX', 'GLUCOSE_MEAN', 'DIASBP_MEAN', 'RESPRATE_MEAN',
         'SYSBP_MEAN', 'HR_MEAN', 'HR_MAX', 'FUTURE_READMIT']]
    # return labelled data with all the important features
    return final_merged_with_admissions_labels_sel_features

# method for remaining preprocessing on the numerical features
def scale_and_impute():
    final_merged_with_admissions_labels_sel_features = assign_Labels()
    final_merged_with_admissions_labvalues = final_merged_with_admissions_labels_sel_features[
        ['ANCHOR_AGE', 'MAGNESIUM_MEAN', 'MAGNESIUM_MAX',
         'PLATELETS_MEAN', 'PLATELETS_MIN', 'PLATELETS_MAX', 'CALCIUM_MEAN',
         'CALCIUM_MIN', 'CALCIUM_MAX', 'UREA_N_MEAN', 'UREA_N_MIN', 'UREA_N_MAX',
         'ALBUMIN_MEAN', 'ALBUMIN_MAX', 'GLUCOSE_MEAN', 'DIASBP_MEAN', 'RESPRATE_MEAN',
         'SYSBP_MEAN', 'HR_MEAN', 'HR_MAX', 'FUTURE_READMIT']]

    print('final_merged_with_admissions_labvalues-->')
    print(final_merged_with_admissions_labvalues.info())

    # Imputing/ filling in of all the numerical values using KNN Imputer wherever it is null
    imputer = KNNImputer(n_neighbors=5)
    final_merged_with_admissions_labvalues_df = pd.DataFrame(
        imputer.fit_transform(final_merged_with_admissions_labvalues),
        columns=final_merged_with_admissions_labvalues.columns)

    print('final_merged_with_admissions_labvalues_df-->')
    print(final_merged_with_admissions_labvalues_df.info())

    final_merged_with_admissions_labvalues_df = pd.concat([final_merged_with_admissions_labels_sel_features[
                                                               ['SUBJECT_ID', 'GENDER', 'HADM_ID', 'PRO_CODE', 'CODE_ICD', 'NDC',
                                                                'DOSES_PER_24_HRS', 'ADMITTIME', 'DISCHTIME']],
                                                           final_merged_with_admissions_labvalues_df], axis=1)

    # To remove outliers
    outliers_remove_final_merged_with_admissions_labvalues_df = final_merged_with_admissions_labvalues_df[
        ['ANCHOR_AGE', 'MAGNESIUM_MEAN', 'MAGNESIUM_MAX',
         'PLATELETS_MEAN', 'PLATELETS_MIN', 'PLATELETS_MAX', 'CALCIUM_MEAN',
         'CALCIUM_MIN', 'CALCIUM_MAX', 'UREA_N_MEAN', 'UREA_N_MIN', 'UREA_N_MAX',
         'ALBUMIN_MEAN', 'ALBUMIN_MAX', 'GLUCOSE_MEAN', 'DIASBP_MEAN',
         'RESPRATE_MEAN', 'SYSBP_MEAN', 'HR_MEAN', 'HR_MAX']]

    # From here till line 380 is related to removal of outliers and their corresponding subjects from the dataset
    scaled_x_df_num = pd.DataFrame(
        StandardScaler().fit_transform(outliers_remove_final_merged_with_admissions_labvalues_df),
        columns=outliers_remove_final_merged_with_admissions_labvalues_df.keys())

    outliers = []

    for feature in scaled_x_df_num.keys():
        # Calculate Q1 (15th percentile of the data) for the given feature
        Q1 = np.percentile(scaled_x_df_num[feature], 15)

        # Calculate Q3 (85th percentile of the data) for the given feature
        Q3 = np.percentile(scaled_x_df_num[feature], 85)

        # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
        step = 1.5 * (Q3 - Q1)

        outliers_per_feature = scaled_x_df_num[
            ~((scaled_x_df_num[feature] >= Q1 - step) & (scaled_x_df_num[feature] <= Q3 + step))]
        outliers = outliers + list(outliers_per_feature.index)

        outliers.sort()
        outliers = list(set(outliers))

    Subject_Ids = []

    for i in outliers:
        Subject_Ids.append(final_merged_with_admissions_labels_sel_features.loc[i, 'SUBJECT_ID'])

    Subject_Ids = set(Subject_Ids)

    # Remove all subjects whos even one admission was removed as part of outliers
    final_preprocessed_list = final_merged_with_admissions_labvalues_df[
        ~final_merged_with_admissions_labvalues_df['SUBJECT_ID'].isin(Subject_Ids)]
    final_preprocessed_list = final_preprocessed_list.reset_index()
    final_preprocessed_list = final_preprocessed_list.sort_values(['SUBJECT_ID', 'ADMITTIME'])

    final_preprocessed_list_scale = final_preprocessed_list[
        ['ANCHOR_AGE', 'MAGNESIUM_MEAN', 'MAGNESIUM_MAX',
         'PLATELETS_MEAN', 'PLATELETS_MIN', 'PLATELETS_MAX', 'CALCIUM_MEAN',
         'CALCIUM_MIN', 'CALCIUM_MAX', 'UREA_N_MEAN', 'UREA_N_MIN', 'UREA_N_MAX',
         'ALBUMIN_MEAN', 'ALBUMIN_MAX', 'GLUCOSE_MEAN', 'DIASBP_MEAN',
         'RESPRATE_MEAN', 'SYSBP_MEAN', 'HR_MEAN', 'HR_MAX']]

    print("final_preprocessed_list_scale: {}".format(final_preprocessed_list_scale.info()))

    # Normalize data for all numerical features
    final_preprocessed_list_scale = pd.DataFrame(preprocessing.normalize(final_preprocessed_list_scale),
                                                 columns=final_preprocessed_list_scale.columns)

    # final preprocessed data
    final_preprocessed_list_scale = pd.concat([final_preprocessed_list[
                                                   ['SUBJECT_ID', 'GENDER','HADM_ID', 'PRO_CODE', 'CODE_ICD', 'NDC',
                                                    'DOSES_PER_24_HRS', 'ADMITTIME', 'DISCHTIME', 'FUTURE_READMIT']],
                                               final_preprocessed_list_scale], axis=1)
    final_preprocessed_list_scale = final_preprocessed_list_scale[
        ['SUBJECT_ID', 'GENDER', 'HADM_ID', 'PRO_CODE',
         'CODE_ICD', 'NDC', 'DOSES_PER_24_HRS', 'ANCHOR_AGE', 'MAGNESIUM_MEAN',
         'MAGNESIUM_MAX', 'PLATELETS_MEAN', 'PLATELETS_MIN', 'PLATELETS_MAX',
         'CALCIUM_MEAN', 'CALCIUM_MIN', 'CALCIUM_MAX', 'UREA_N_MEAN',
         'UREA_N_MIN', 'UREA_N_MAX', 'ALBUMIN_MEAN', 'ALBUMIN_MAX',
         'GLUCOSE_MEAN', 'DIASBP_MEAN', 'RESPRATE_MEAN', 'SYSBP_MEAN', 'HR_MEAN','HR_MAX',
         'ADMITTIME','DISCHTIME','FUTURE_READMIT']]

    patient_codes_cleaned_path = os.path.join(features_base_path, 'patient_codes_cleaned_step2.pickle')
    final_preprocessed_list_scale.to_pickle(patient_codes_cleaned_path)
    print(len(final_preprocessed_list_scale.SUBJECT_ID.unique()))
# return final_preprocessed_list_scale


scale_and_impute()