'''
    This is a script that generates the required features for training the Gate model
    Features are generated and stored under features directory
'''

import os
import pandas as pd
import pickle5

#from Preprocess_Step_1 import combine_codes
from global_matrix import build_save_event_matrix
from patientDictCreation import create_save_patient_dict, save_medication_labels
#from Preprocess_Step_2_LabEventsAndChart import scale_and_impute


def finalPreproccessing():

    features_base_path = r'preprocess1.0/Processed_files'

    patient_codes_cleaned_path = os.path.join(features_base_path, 'patient_codes_cleaned_step2.pickle') # All Codes grouped by Patient/Admission
    M_matrix_path = os.path.join(features_base_path, 'all_codes_pmi_matrix.pickle')                     # PMI matrix for fequency of coocurrence
    event_location_dict_path = os.path.join(features_base_path, 'event_location_dict.pickle')           # Dictionary mapping code names to Ids
    med_labels_path = os.path.join(features_base_path, 'medications_labels.pickle')                     # List of medications found in dataset
    patient_dict_path = os.path.join(features_base_path, 'patient_dict.pickle')                         # Dictionary of codes by admission for training

    '''
    STEP 1: Create Grouped and Filtered Dataframe of format-
        Patient_Id      Admission_id        Diagnosis_ICD_Codes     Meds_ATC_Codes     Procedure_Codes
        1               1                   []                      []                  []
        1               2                   []                      []                  []
        2               1                   []                      []                  []
    Store in 'patient_codes_cleaned.pickle'
    '''

    #scale_and_impute()

    '''
    STEP 2: Use above dataframe to create dictionary of all event codes 
            and PMI Matrix corresponding to all codes based on co-occurrence in single admission
    Event Dictionary stored in 'event_location_dict.pickle'
    Complete PMI matrix for all codes stored in 'all_codes_pmi_matrix.pickle'
    '''

    build_save_event_matrix(patient_codes_cleaned_path, M_matrix_path, event_location_dict_path)


    '''
    STEP 3: Save Medications Labels
    '''

    save_medication_labels(patient_codes_cleaned_path, med_labels_path)


    '''
    STEP 4: Using above dataframe and PMI matrix, we construct a patient Dictionary
        Patient: {
            [
                [All_codes], [Subset of PMI Matrix]                    // Admission 1
            ],     
            [...]      // Admission 2
            .....
            [   [Codes_without_meds], [Corresponding pmi subset]   
            ]      // last admission
        }, drug recommendation map (Y output for training)
    '''

    create_save_patient_dict(patient_codes_cleaned_path, M_matrix_path, patient_dict_path, event_location_dict_path, med_labels_path)



finalPreproccessing()