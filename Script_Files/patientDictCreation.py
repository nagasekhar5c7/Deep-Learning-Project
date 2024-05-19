'''
Methods for Building Patient Dictionary with event admission history
Result used as input for GATE model
'''

import pickle as pickle5
import numpy as np
from tqdm.auto import tqdm
import torch
from tqdm.auto import tqdm
from datetime import date


#feature_list = ['PRO_CODE','CODE_ICD', 'NDC','GENDER','ANCHOR_AGE']
feature_list = ['PRO_CODE','CODE_ICD', 'NDC','GENDER']
# feature_list = ['PRO_CODE','CODE_ICD', 'NDC']

#hyp =   ['4010','4011','4019','40200','40201','40210','40211','40290','40291','I10','I11','I110','I119','I12','I120','I129','I13','I130','I131','I1310','I1311','I132',
#        'I15','I150','I151','I152','I158','I159','I16','I160','I161','I169']

#con =   ['4280','39891','4281','42820','42821','42822','42823','42830','42831','42832','42833','42840','42841','42842','42843','I50', 'I501', 'I502', 'I5020', 'I5021', 
#        'I5022', 'I5023', 'I503', 'I5030', 'I5031', 'I5032', 'I5033', 'I504', 'I5040', 'I5041', 'I5042', 'I5043', 'I508', 'I5081', 'I50810', 'I50811', 'I50812', 
#        'I50813', 'I50814', 'I5082', 'I5083', 'I5084', 'I5089', 'I509']

#afib =  ['I48','I480','I481','I4811','I4819','I4820','I4821','I489','I4891','42731']

#cor_art = ['41400','41401','41402','41403','41404','41405','41406','41407',' 4143','4144',' I2583',' I2584','I251',' I2510','I2511','I25110','I25111','I25118','I25119']
#cor_art = ['41400', '41401', '41402', '41403', '41404', '41405', '41406', '41407', ' 4143', '4144',
#           'I25', 'I251', 'I2510', 'I2511', 'I25110', 'I25111', 'I25112', 'I25118', 'I25119', 'I252', 'I253', 'I254', 'I2541',
#           'I2542', 'I255', 'I256', 'I257', 'I2570', 'I25700', 'I25701', 'I25702', 'I25708', 'I25709', 'I2571', 'I25710',
#          'I25711', 'I25712', 'I25718', 'I25719', 'I2572', 'I25720', 'I25721', 'I25722', 'I25728', 'I25729', 'I2573', 'I25730',
#           'I25731', 'I25732', 'I25738', 'I25739', 'I2575', 'I25750', 'I25751', 'I25752', 'I25758', 'I25759', 'I2576', 'I25760',
#           'I25761', 'I25762', 'I25768', 'I25769', 'I2579', 'I25790', 'I25791', 'I25792', 'I25798', 'I25799', 'I258', 'I2581',
#           'I25810', 'I25811', 'I25812', 'I2582', 'I2583', 'I2584', 'I2589', 'I259']

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

def get_label_idxs(meds, med_location_dict):
    return [med_location_dict[med] for med in meds]


def build_dynamic_co_occurence_graph(
        patient_id, data, m_matrix, event_location_dict):
    """
    Building dynamic co occurrence graph for an individual patient
    according to a single admission.
    """

    list_timesteps = []
    A_dict_list = []
    # label_idxs = []
    data = data[data.SUBJECT_ID == patient_id]
    n = data.shape[0]
    # whole_idx = [i for i in range(m_matrix.shape[0])]

    admission_num = 0
    for idx, row in data.iterrows():
        admission_num += 1

        # if admission_num==n:
        #    ev_list = diag + proc
        # else:
        #    ev_list = ndc + diag + proc

        #ev_list = []
        #for col in feature_list:
            #print(row[col])
            #print(type(row[col]))
            #ev_list.append(row[col])
        #print(ev_list)
        ev_list = [[row[col]] if col == "GENDER" else row[col] for col in feature_list]
        ev_list = list(np.concatenate(ev_list).flat)

        # ev_idxs = [event_location_dict[ev] for ev in ev_list]
        evs = []
        for ev in ev_list:
            if ev in event_location_dict:
                evs.append(ev)
        ev_list = evs
        dynamic_single_step = np.zeros((len(ev_list), len(ev_list)))
        A_dict = {}
        for idx1, item in enumerate(ev_list):
            loc_item1 = event_location_dict[item]
            for idx2, item2 in enumerate(ev_list):
                loc_item2 = event_location_dict[item2]

                dynamic_single_step[idx1, idx2] = m_matrix[loc_item1, loc_item2]

                A_dict[item] = idx1

        list_timesteps.append(dynamic_single_step)
        A_dict_list.append(A_dict)

    return list_timesteps, A_dict_list


def make_patient_H0s(patient_list, recs_add, M_matrix, event_location_dict, labels_dict):
    '''
    creates a dictionary, key: patient_id, value: list of lists of lists, each outer list is a patient,
    first nested list is from the 0th up until the nth admission,
    second nested list is
    [co_occurence_matrix, code indices, co_occurence matrix without medications, indices without medications, ground truth label]
    for the ith admission
    '''
    hyp_set = set(hyp)
    con_set = set(con)
    afib_set = set(afib)
    cor_art_set = set(cor_art)

    data_dict = {}
    for patient_id in tqdm(patient_list):
        A, A_dict = build_dynamic_co_occurence_graph(patient_id, recs_add, M_matrix, event_location_dict)
        data = recs_add[recs_add.SUBJECT_ID == patient_id].reset_index()
        Ht_list = []
        for i, A_i in enumerate(A):  
           # Admissions
            y = data.iloc[i, :]['FUTURE_READMIT']
            y = torch.as_tensor(y, dtype=torch.int, device=None)
            ndc = data.iloc[i, :]['NDC']
            ev_list = []
            word_list = []
            for col in feature_list:
                #ev_list += list(str(data.iloc[i, :][col]))
                if col == "GENDER":
                    ev_list += [data.iloc[i, :][col]]
                else:
                    ev_list += list(data.iloc[i, :][col])
            evs = []
            for ev in ev_list:
                if ev in event_location_dict:
                    evs.append(ev)
            ev_list = evs
            ev_idxs = [event_location_dict[ev] for ev in ev_list]
            # H0 = embeddings(torch.tensor(ev_idxs))

            '''A_i_test, test_ev_idxs, y = None, None, None
            if i > 0:
                test_ev_list = []
                test_word_list = []
                for col in feature_list[0:-1]:  # because we're excluding ndc
                    test_ev_list += list(data.iloc[i, :][col])
                evs = []
                for ev in test_ev_list:
                    if ev in event_location_dict:
                        evs.append(ev)
                test_ev_list = evs
                test_ev_idxs = torch.tensor([event_location_dict[ev] for ev in test_ev_list])
                A_i_test = torch.tensor(A_i[0:len(ev_list) - len(ndc), 0:len(ev_list) - len(ndc)])
                class_labels = get_label_idxs(ndc, labels_dict)
                y = torch.zeros(len(labels_dict))
                y[class_labels] = 1
            else:
                y = 0'''

            #print(data.info())

            Ht_list.append([torch.tensor(A_i), torch.tensor(ev_idxs), torch.tensor(data.iloc[i, 8:28]),
                            torch.tensor(data.iloc[i, -3].timestamp()), torch.tensor(data.iloc[i, -2].timestamp()), y])

        data_dict[patient_id] = Ht_list
    return data_dict


def create_save_patient_dict(recs_add_path, M_matrix_path, patient_dict_path, event_location_dict_path, meds_path):
    '''
    Read Recurring Admission data and event Matrix
    Construct patient Dictionary and Medications List
    and save to file
    '''
    with open(recs_add_path, 'rb') as file:
        recs_add = pickle5.load(file)

    with open(event_location_dict_path, 'rb') as file:
        event_location_dict = pickle5.load(file)

    with open(M_matrix_path, 'rb') as f:
        M_matrix = pickle5.load(f)

    with open(meds_path, 'rb') as f:
        med_labels = pickle5.load(f)

    patient_list = recs_add.SUBJECT_ID.unique()
    patient_dict = make_patient_H0s(patient_list, recs_add, M_matrix, event_location_dict, med_labels)

    with open(patient_dict_path, 'wb') as f:
        pickle5.dump(patient_dict, f, pickle5.HIGHEST_PROTOCOL)


def save_medication_labels(patient_codes_cleaned_path, meds_path):
    '''
    Create medications Labels Dictionary and save to file
    '''
    with open(patient_codes_cleaned_path, 'rb') as file:
        combined_codes = pickle5.load(file)

    med_labels = sorted(list(set([med for p in combined_codes['NDC'] for med in p])))
    labels_dict = {k: v for v, k in enumerate(med_labels)}

    with open(meds_path, 'wb') as f:
        pickle5.dump(labels_dict, f, pickle5.HIGHEST_PROTOCOL)