import os
import os.path as ops
import json
import pickle
import numpy as np
import pandas as pd



mimic_base_path = r'project/mimiciv1.0'
inputs_base_path = r'project/inputs'
features_base_path = r'preprocess1.0/Processed_files'


PROC_FILE = os.path.join(mimic_base_path, 'procedures_icd.csv.gz')        # ICD codes per
MED_FILE = os.path.join(mimic_base_path, 'prescriptions.csv.gz')          # NDC codes per admission/patient
DIAG_FILE = os.path.join(mimic_base_path, 'diagnoses_icd.csv.gz')         # ICD codes per admission/patient
LAB_EVENTS = os.path.join(mimic_base_path, 'labevents.csv.gz')
PATIENT_FILE = os.path.join(mimic_base_path, 'patients.csv.gz')
ICU_FILE = os.path.join(mimic_base_path, 'icustays.csv.gz')
CHART_EVENTS = os.path.join(mimic_base_path, 'chartevents.csv.gz')
ADM_FILE = os.path.join(mimic_base_path, 'admissions.csv.gz')


ndc2atc_file = os.path.join(inputs_base_path, 'ndc2atc_level4.csv' )
cid_atc = os.path.join(inputs_base_path, 'drug-atc(1).csv')
ndc2rxnorm_file = os.path.join(inputs_base_path, 'ndc2rxnorm_mapping(1).txt')
# DDI_FILE = os.path.join(inputs_base_path, 'drug-DDI.csv')

icd9_2_10 = json.loads(open(os.path.join(inputs_base_path, 'icd9_2_10.json'), 'r').read())
cardiac_codes_10 = json.loads(open(os.path.join(inputs_base_path, 'cardiac_codes_10.json'), 'r').read())


def process_procedure():
    """
    Preprocess procedure file to drop extra columns, remove duplicates and sort
    by Subject, Admission and sequence.
    """
    pro_pd = pd.read_csv(PROC_FILE, dtype={'icd9_code':'category'})
    pro_pd.columns = pro_pd.columns.str.upper()
    DROP_COLS = ['SEQ_NUM']
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
    pro_pd.drop(columns=DROP_COLS, inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)
    return pro_pd

def process_admissions():
    admissions_df = pd.read_csv(ADM_FILE)
    admissions_df = admissions_df.drop(columns = ['admission_location', 'discharge_location', 'language', 'edregtime', 'edouttime', 'deathtime']) 
    return admissions_df
                                    

def process_med():
    """
    Function to clean and preprocess medications
    1. Drops extra columns and removes duplicates
    2. Sorts by subject ID, admission ID and time
    3. Finds medications prescribed in first 60 days of admission
    """
    med_pd = pd.read_csv(MED_FILE, dtype={'ndc': 'category'})
    med_pd.columns = med_pd.columns.str.upper()
    DROP_COLS = ['DRUG_TYPE', 'GSN', 'PROD_STRENGTH', 'DOSE_VAL_RX', \
                 'DOSE_UNIT_RX', 'FORM_VAL_DISP', 'FORM_UNIT_DISP', \
                 'FORM_UNIT_DISP', 'ROUTE', 'DRUG', 'FORM_RX', 'STOPTIME']

    med_pd.drop(columns=DROP_COLS, axis=1, inplace=True)
    med_pd.drop(index=med_pd[med_pd['NDC'] == '0'].index, axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['STARTTIME'] = pd.to_datetime(med_pd['STARTTIME'], format='%Y-%m-%d %H:%M:%S')
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'STARTTIME'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)

    # Drop extra columns and duplicates
    med_pd.drop(columns=['PHARMACY_ID'], inplace=True)
    med_pd.drop_duplicates(inplace=True)
    return med_pd.reset_index(drop=True)


def process_diag():
    """
    Preprocess diagnoses
    1. Droping nulls and duplicates
    2. Sorting
    3. Get cardiac diagnosses
    """
    diag_pd = pd.read_csv(DIAG_FILE)
    diag_pd.columns = diag_pd.columns.str.upper()
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['SEQ_NUM'], inplace=True)  # remove row id
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)

    def process_codes(code_string):
        return code_string.split(' ')[0]

    def get_cardiac_diag(df_diag):
        """
        Getting diagnoses that are cardiac  related
        """
        df_diag['ICD_CODE'] = df_diag['ICD_CODE'].apply(process_codes)
        cardiac_9 = list(icd9_2_10.keys())
        cardiac_10 = list(icd9_2_10.values())
        codes_in_9 = (df_diag.ICD_CODE.isin(cardiac_9))
        codes_in_10 = (df_diag.ICD_CODE.isin(cardiac_9))
        df_diag = df_diag[(codes_in_10) | (codes_in_9)]
        return df_diag

    diag_pd['ICD_CODE'] = diag_pd['ICD_CODE'].apply(process_codes)
    
    # Creating a new column severity
    for i in diag_pd['ICD_CODE']:
        if i.endswith('00'):
            print("Asymptomatic, no treatment needed at this time.")
            diag_pd['severity'] = 0
        elif i.endswith('01'):
            print("Symptioms well controlled with current therapy")
            diag_pd['severity'] = 1
        elif i.endswith('02'):
            print("Symptoms controlled with difficulty, affecting daily functioning; patinet needs ongoing monitoring")
            diag_pd['severity'] = 2
        elif i.endswith('03'):
            print("Symptoms poorly controlled, patient needs frequent adjustment in treamtent and dose monitoring")
            diag_pd['severity'] = 3
        elif i.endswith('04'):
            print("Symptoms poorly controlled, history of rehospitalizations.")
            diag_pd['severity'] = 4
        else:
            continue

    return diag_pd.reset_index(drop=True)

def process_patient():
    pati_pd = pd.read_csv(PATIENT_FILE)
    pati_pd.columns = pati_pd.columns.str.upper()
    
    DROP_COLS = ['DOD','ANCHOR_YEAR','ANCHOR_YEAR_GROUP']

    pati_pd.drop(columns=DROP_COLS, axis=1, inplace=True)
    pati_pd.fillna(method='pad', inplace=True)
    pati_pd.dropna(inplace=True)
    pati_pd.drop_duplicates(inplace=True)
    return pati_pd.reset_index(drop=True)

def ndc2atc(med_pd):
    with open(ndc2rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())
    """
    Mapping from one type of medication to another
    1. Mapping to ATC from NDC
    2. Drop extra columns
    3. Finding ATC level 3 code
    """
    med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True)

    rxnorm2atc = pd.read_csv(ndc2atc_file)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR', 'MONTH', 'NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_pd.drop(index=med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)

    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])
    med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True)
    med_pd = med_pd.rename(columns={'ATC4': 'NDC'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: x[:4])  # atc level 4 code
    # # turn NDC to ACT code
    # med_pd.insert(med_pd.shape[1],'ACT',med_pd['NDC'])
    # med_pd.drop(columns=['NDC'], axis=1, inplace=True)
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd


def combine_codes():
    proc = process_procedure()
    med_pd = process_med()
    med_pd = ndc2atc(med_pd)
    diag_pd = process_diag()
    patient_pd = process_patient()

    med_diag = med_pd.merge(diag_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    card_codes = list(set(list(cardiac_codes_10.keys()))) + list(set(list(icd9_2_10.keys())))
    cond_1 = med_diag.ICD_CODE.isin(card_codes)
    cond_2 = med_diag.ICD_CODE.str.startswith(tuple(card_codes))
    mdc = med_diag[(cond_1) | (cond_2)]
    combined_key = mdc[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    # Merging combined key on all medications/ diagnoses/ procedures
    meds_new = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    diag_new = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    proc_new = proc.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pati_new = patient_pd.merge(combined_key, on=['SUBJECT_ID'], how='inner')


    print("##################### 1 Merging successfully launched ######################")
    
    def combine_icd_codes(version, code):
        return str(version) + '.' + str(code)

    def gender2code(gender):
        code = 0 if gender=='M' else 1
        return code

###################################### Pro code, Patient, Dialgose #####################################################
    # Applying code changes to diagnosis column to create new diagnosis column and dropping old columns
    diag_new.insert(diag_new.shape[1],'CODE_ICD',diag_new['ICD_CODE'])
    diag_new['CODE_ICD'] = diag_new.apply(lambda x: combine_icd_codes(x['ICD_VERSION'], x['ICD_CODE']), axis=1)
    diag_new.drop(columns=['ICD_CODE', 'ICD_VERSION'], inplace=True)

    # Grouping by hospital admissions and patient so that each row is an admission
    diag_new = diag_new.groupby(by=['SUBJECT_ID', 'HADM_ID'])['CODE_ICD'].unique().reset_index()

    # Grouping by hospital admissions and patient so that each row is an admission
    proc_new = proc_new.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ICD_CODE'].unique().reset_index()\
                    .rename(columns={'ICD_CODE':'PRO_CODE'})

    pati_new['GENDER'] = pati_new.apply(lambda x: gender2code(x['GENDER']),axis=1)
    #pati_new['GENDER'] = pati_new.apply(lambda x: np.array(x['GENDER']),axis=1)


    def fusion_NDC_Dosage(NDC,Dosage):
        return str(NDC)+'-'+str(Dosage)

    def seperate_NDC_Dosage(NDC,Dosage,NDC_Dosage_List):
        NDC_new = []
        Dosage_new = []
        for item in NDC_Dosage_List:
            Code = item.split('-')
            NDC_new.append(Code[0])
            Dosage_new.append(round(float(Code[1]),2))
        return NDC_new, Dosage_new


######################################### Drug and Dosage #############################################################

    # calculating the average value of the drugs
    dosage_avg = meds_new.groupby(by=['SUBJECT_ID', 'HADM_ID','NDC']).agg({'DOSES_PER_24_HRS':'mean'})
    drug_dosage_new = meds_new.merge(dosage_avg,on=['SUBJECT_ID', 'HADM_ID','NDC'],how='inner')

    # merge the drug and its dosage
    drug_dosage_new['DOSES_PER_24_HRS_y'] = drug_dosage_new.apply(lambda x: fusion_NDC_Dosage(x['NDC'], x['DOSES_PER_24_HRS_y']), axis=1) 

    drug_dosage_new = drug_dosage_new.groupby(by=['SUBJECT_ID', 'HADM_ID'])['DOSES_PER_24_HRS_y'].unique().reset_index()# filter duplicates

    drug_dosage_new.insert(drug_dosage_new.shape[1],'DOSES_PER_24_HRS',drug_dosage_new['DOSES_PER_24_HRS_y'])

    drug_dosage_new.insert(drug_dosage_new.shape[1],'NDC',drug_dosage_new['DOSES_PER_24_HRS_y'])

    drug_dosage_new[['NDC','DOSES_PER_24_HRS']] = drug_dosage_new.apply(lambda x: \
                                                            seperate_NDC_Dosage(x['NDC'],x['DOSES_PER_24_HRS'],x['DOSES_PER_24_HRS_y']),\
                                                           axis=1,result_type='expand') #sperate NDC and dosage

    drug_dosage_new.drop(columns=['DOSES_PER_24_HRS_y'],inplace=True)


    # Merging and combining them into a single dataset
    # proc_new
    # diag_new
    # pati_new
    # drug_dosage_new
    combined = proc_new.merge(diag_new, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined = combined.merge(pati_new, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined = combined.merge(drug_dosage_new, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    #print("combined: {}".format(combined.info()))
    
    patient_codes_cleaned_path = os.path.join(features_base_path, 'patient_codes_cleaned_step1.pickle')
    with open(patient_codes_cleaned_path, 'wb') as file:
        pickle.dump(combined, file, protocol=pickle.HIGHEST_PROTOCOL)
   
#    return combined


combine_codes()

