#%%
import pandas as pd
import os
import re


def create_or_load_dataset(data_path, files,features_output_path,output_num,create=False):
    # Initialize common_patients with the first file
    first_file = os.path.join(data_path, files[0])
    df = pd.read_csv(first_file, usecols=["PATIENT"])
    common_patients = set(df["PATIENT"].unique())
    print(f'Reading from Synthea output {output_num}')
    # Iterate over remaining files and find the intersection
    for file in files[1:]:
        print(file)
        file_path = os.path.join(data_path, file)
        if file=="patients.csv":
            df = pd.read_csv(f"{data_path}{file}", usecols=["ID"])  # Load only the 'PATIENT' column
            patient_ids = set(df["ID"].unique())
            continue
        df = pd.read_csv(file_path, usecols=["PATIENT"])
        patient_ids = set(df["PATIENT"].unique())
        # Find common patients
        common_patients &= patient_ids
    # Convert to sorted list
    common_patients = sorted(common_patients)
    print(f"Found {len(common_patients)} common patients.")
    # Read conditions.csv and filter for Atopic Dermatitis
    conditions_df = pd.read_csv(os.path.join(data_path, "conditions.csv"))
    conditions_df = conditions_df[conditions_df["PATIENT"].isin(common_patients)]
    # Identify patients with Atopic Dermatitis
    ad_patients = set(conditions_df[conditions_df["CODE"]==24079001]["PATIENT"].unique())
    print(f'Number of patients with Atopic Dermatitis: {len(ad_patients)}. ({len(ad_patients)*100/len(common_patients)})')
    # Assign labels
    labels = {pid: 1 if pid in ad_patients else 0 for pid in common_patients} #2705
    
    def create_feature_EHR(relevant_features,df,time_col_name='START',target_col='DESCRIPTION') :   
        # List to store the allergy feature data
        data = []
        # Loop through the common patients and check if they have any of the relevant allergies
        for patient in common_patients:
            for feature in relevant_features:
                # Filter the allergies data for the patient
                patient_data = df[df['PATIENT'] == patient]
                
                # Check if the patient has the allergy
                has_feature = 1 if any(patient_data[target_col] == feature) else 0
                
                # Get the 'START' column as time
                if has_feature == 1:
                    time_start = patient_data[patient_data[target_col] == feature][time_col_name].min()
                else:
                    time_start = None  # If no allergy, no start time

                # Add the allergy feature to the list
                data.append({
                    'patient_id': patient,
                    'time': time_start,
                    'feature_name': feature,
                    'value': has_feature
                })

        # Convert the list into a DataFrame
        features_df = pd.DataFrame(data)
        # Ensure that the DataFrame has the right format (repeated patient_id for each allergy)
        # Repeat for all patients and allergies
        features_df = features_df.dropna(subset=["time"])  # drop rows without valid time
        # Sort by patient_id and time
        features_df.sort_values(by=["patient_id", "time"], inplace=True)
        return features_df
    
    if create:
        # allergies        
        csv_allergies = pd.read_csv(os.path.join(data_path, "allergies.csv"))
        csv_allergies=csv_allergies[csv_allergies['PATIENT'].isin(common_patients)]
        allergy_features = csv_allergies['DESCRIPTION'].unique()
        allergies_features_df = create_feature_EHR(allergy_features,csv_allergies,time_col_name='START') 
        allergies_features_df.to_csv(os.path.join(features_output_path, "allergies_features.csv"),index=False)
        # careplans
        care_plans = pd.read_csv(os.path.join(data_path, "careplans.csv"))
        care_plans = care_plans[care_plans['PATIENT'].isin(common_patients)]
        careplans_features = [
            "Skin condition care",
            "Application of moisturizer to skin",
            "Allergy education",
            "Food allergy diet",
            "Stress management",
            "Allergic disorder monitoring",
            "Allergen immunotherapy drugs Band 1",
            "Behavior to prevent infection",
            "Wound care",
            "Dressing change management",
            "Psychological assessment",
            "Counseling"
        ]
        care_plans_rep = care_plans[care_plans['DESCRIPTION'].isin(careplans_features)]
        careplans_features_df = create_feature_EHR(careplans_features,care_plans_rep,time_col_name='START',target_col='DESCRIPTION') 
        careplans_features_df.to_csv(os.path.join(features_output_path, "careplans_features.csv"),index=False)
        # medications
        medications = pd.read_csv(os.path.join(data_path,"medications.csv"))
        medications = medications[medications['PATIENT'].isin(common_patients)]
        # Filter for common patients
        medications = medications[medications["PATIENT"].isin(common_patients)]
        # Expanded relevant medications list
        relevant_meds = relevant_meds = [
            "cetirizine hydrochloride 5 MG Oral Tablet",
            "diphenhydrAMINE Hydrochloride 25 MG Oral Tablet",
            "predniSONE 5 MG Oral Tablet",
            "Hydrocortisone 10 MG/ML Topical Cream",
            "Fexofenadine hydrochloride 30 MG Oral Tablet",
            "Fexofenadine hydrochloride 60 MG Oral Tablet",
            "Loratadine 5 MG Chewable Tablet",
            "Acetaminophen 325 MG Oral Tablet",
            "Acetaminophen 300 MG / HYDROcodone Bitartrate 5 MG [Vicodin]",
            "Acetaminophen 325 MG / HYDROcodone Bitartrate 7.5 MG Oral Tablet",
            "Allergic disorder monitoring",
            "Allergen immunotherapy drugs Band 1"
        ]
        # Function to extract medication name and dosage
        def extract_medication_info(description):
            description = description.lower().strip()
            # Find the medication name
            matched_med = None
            for med in relevant_meds:
                if re.search(rf'\b{re.escape(med.lower())}\b', description):
                    matched_med = med
                    break
            if not matched_med:
                return None, None  # Skip if no relevant med found
            # Extract numeric dosage (handles cases like "250 MG", "0.25 MG", "200 ACTUAT")
            match = re.search(r'(\d+(\.\d+)?)\s*(MG|ML|ACTUAT|Tablets|Capsules)?', description)
            dosage = float(match.group(1)) if match else None
            return matched_med, dosage

        # Apply function to extract name & dosage
        medications[["feature_name", "value"]] = medications["DESCRIPTION"].apply(lambda x: pd.Series(extract_medication_info(str(x))))
        # Drop rows where no medication was extracted
        medications = medications.dropna(subset=["feature_name", "value"])
        # Select required columns
        medications_features = medications[["PATIENT", "START", "feature_name", "value"]].rename(columns={"PATIENT": "patient_id", "START": "time"})
        medications_features.to_csv(os.path.join(features_output_path, "medications_features.csv"),index=False)

        #encounters
        encounters = pd.read_csv(os.path.join(data_path,"encounters.csv"))
        encounters = encounters[encounters['PATIENT'].isin(common_patients)]
        encounters_features = relevant_encounters = [
            "Outpatient Encounter",
            "Encounter for symptom",
            "Encounter for problem",
            "Asthma follow-up",
            "Emergency hospital admission for asthma",
            "Allergic disorder initial assessment",
            "Allergic disorder follow-up assessment",
            "Outpatient procedure",
            "Patient encounter procedure",
            "Emergency Encounter",
            "Office Visit"
        ]
        #encounters['DESCRIPTION'].unique().tolist()
        encounters_features_df = create_feature_EHR(encounters_features,encounters,time_col_name='DATE') 
        encounters_features_df.to_csv(os.path.join(features_output_path, "encounters_features.csv"),index=False)
        
        # immunization
        immunizations = pd.read_csv(os.path.join(data_path,"immunizations.csv"))
        immunizations = immunizations[immunizations['PATIENT'].isin(common_patients)]
        immunizations_features = immunizations_features = relevant_immunizations = [
            "Hep B adolescent or pediatric",
            "DTaP",
            "Hib (PRP-OMP)",
            "Pneumococcal conjugate PCV 13",
            "IPV",
            "Influenza seasonal injectable preservative free",
            "MMR",
            "Varicella",
            "Hep A ped/adol 2 dose",
            "HPV quadrivalent",
            "Meningococcal MCV4P",
            "Tdap"
        ]#immunizations['DESCRIPTION'].unique().tolist()
        immunizations_features_df = create_feature_EHR(immunizations_features,immunizations,time_col_name='DATE') 
        immunizations_features_df.to_csv(os.path.join(features_output_path, "immunizations_features.csv"),index=False)

        #conditions
        conditions_df = pd.read_csv(os.path.join(data_path, "conditions.csv"))
        conditions_df = conditions_df[conditions_df["PATIENT"].isin(common_patients)]
        relevant_conditions = [
            "Asthma",
            "Childhood asthma",
            "Perennial allergic rhinitis",
            "Seasonal allergic rhinitis",
            "Acute allergic reaction",
            "Chronic sinusitis (disorder)",
            "Acute bacterial sinusitis (disorder)",
            "Chronic intractable migraine without aura",
            "Rheumatoid arthritis",
            "Lupus erythematosus",
            "Chronic pain",
            "Primary fibromyalgia syndrome"
        ]
        conditions_features = create_feature_EHR(relevant_conditions,conditions_df,time_col_name='START')         
        conditions_features.to_csv(os.path.join(features_output_path, "conditions_features.csv"),index=False)

        # observation
        observations = pd.read_csv(os.path.join(data_path, "observations.csv"))
        observations = observations[observations["PATIENT"].isin(common_patients)]
        relevant_observations = [
            "Hemoglobin A1c/Hemoglobin.total in Blood","Body Height",
            "Body Weight","Body Mass Index","Systolic Blood Pressure",
            "Diastolic Blood Pressure","Glucose","Urea Nitrogen", "Creatinine","Calcium",
            "Sodium", "Potassium","Chloride","Carbon Dioxide","Total Cholesterol",
            "Triglycerides", "Low Density Lipoprotein Cholesterol",
            "High Density Lipoprotein Cholesterol","Microalbumin Creatine Ratio",
            "Estimated Glomerular Filtration Rate", "Oral temperature",
            "Peanut IgE Ab in Serum",
            "Walnut IgE Ab in Serum",
            "Codfish IgE Ab in Serum",
            "Shrimp IgE Ab in Serum",
            "Wheat IgE Ab in Serum",
            "Egg white IgE Ab in Serum",
            "Soybean IgE Ab in Serum",
            "Cow milk IgE Ab in Serum",
            "White oak IgE Ab in Serum",
            "Common Ragweed IgE Ab in Serum",
            "Cat dander IgE Ab in Serum",
            "American house dust mite IgE Ab in Serum",
            "Cladosporium herbarum IgE Ab in Serum",
            "Honey bee IgE Ab in Serum",
            "Latex IgE Ab in Serum", "DXA [T-score] Bone density",
            "FEV1/FVC", "Hemoglobin.gastrointestinal [Presence] in Stool by Immunologic method"
        ]
        observations = observations[observations['DESCRIPTION'].isin(relevant_observations)]
        observations_features = observations[['PATIENT','DATE','DESCRIPTION','VALUE']]
        observations_features.rename(columns={'PATIENT':'patient_id','DATE':'time','DESCRIPTION':'feature_name','VALUE':'value'},inplace=True)
        observations_features.to_csv(os.path.join(features_output_path, "observations_features.csv"),index=False)

        # patients
        patients = pd.read_csv(os.path.join(data_path, "patients.csv"), error_bad_lines=False, warn_bad_lines=True)
        patients=patients[patients['ID'].isin(common_patients)]
        # Extract relevant columns
        features_BD = patients[['ID', 'BIRTHDATE']].copy()
        # Rename columns to match the required format
        features_BD['feature_name'] = "Birth Date"
        features_BD['value'] = features_BD['BIRTHDATE']  # Date stored in 'value'
        features_BD['time'] = features_BD['BIRTHDATE']   # Also stored in 'time'
        features_BD['time'] = pd.to_datetime(features_BD['time'])
        # Rename patient ID column
        features_BD.rename(columns={'ID': 'patient_id'}, inplace=True)
        # Select final columns
        features_BD = features_BD[['patient_id', 'time', 'feature_name', 'value']]
        features_BD.to_csv(os.path.join(features_output_path, "features_BD.csv"),index=False)

        # procedures
        procedures = pd.read_csv(os.path.join(data_path, "procedures.csv"))
        procedures = procedures[procedures['PATIENT'].isin(common_patients)]
        procedures_features = [
            "Fracture subluxation of wrist",
            "Rupture of patellar tendon",
            "Laceration of foot",
            "Facial laceration",
            "Fracture of ankle",
            "Fracture of clavicle",
            "Normal pregnancy",
            "Laceration of hand",
            "Laceration of thigh",
            "Closed fracture of hip",
            "Child attention deficit disorder",
            "Bullet wound",
            "Laceration of forearm",
            "Polyp of colon",
            "Recurrent rectal polyp",
            "Tubal pregnancy",
            "Third degree burn",
            "Fracture of forearm",
            "Fracture of rib",
            "Appendicitis",
            "Injury of medial collateral ligament of knee",
            "Injury of anterior cruciate ligament",
            "Injury of tendon of the rotator cuff of shoulder",
            "Secondary malignant neoplasm of colon",
            "Fracture of the vertebral column with spinal cord injury",
            "Primary malignant neoplasm of colon"
        ]
        #procedures['DESCRIPTION'].unique().tolist()
        procedures_features_df = create_feature_EHR(procedures_features,procedures,target_col='REASONDESCRIPTION',time_col_name='DATE') 
        procedures_features_df.to_csv(os.path.join(features_output_path, "procedures_features.csv"),index=False)

        
        #----------------------------------------
        print('Combining features')
        csv_files = [f for f in os.listdir(features_output_path) if f.endswith('.csv')]
        [careplans_features,medications_features,allergies_features_df,observations_features,procedures_features,BirthDate_features,conditions_features,immunizations_features,encounters_features] = [pd.read_csv(os.path.join(features_output_path, file)) for file in csv_files]
        dfs = [pd.read_csv(os.path.join(features_output_path, file)) for file in csv_files]   
        # Concatenate all DataFrames vertically
        combined_features_df = pd.concat(dfs, ignore_index=True)
        # Convert labels dict keys (patient IDs) into a list in the given order
        patient_order = list(labels.keys())
        # Create a categorical column in the DataFrame to preserve the custom order
        combined_features_df["patient_id"] = pd.Categorical(combined_features_df["patient_id"], categories=patient_order, ordered=True)
        # Sort the DataFrame based on the custom patient order
        combined_features_df = combined_features_df.sort_values(by=["patient_id", "time"])  
        print('Filtering Data')  
        #filter
        # Get patients with NaN values in the 'value' column
        patients_with_nan = combined_features_df[combined_features_df["value"].isna()]["patient_id"].unique()
        print(f'remove {len(patients_with_nan)} patients because of nan.')

        # Get patients that have the "Birth Date" feature
        patients_with_birthdate = combined_features_df.loc[
            combined_features_df["feature_name"] == "Birth Date", "patient_id"
        ].unique()
        # Find patients to remove (either have NaN values or lack Birth Date)
        patients_to_remove = set(patients_with_nan) | set(labels.keys()) - set(patients_with_birthdate)
        print(f'remove {len(patients_to_remove)} patients.')
        # Remove these patients from combined_features_df
        combined_features_df = combined_features_df[~combined_features_df["patient_id"].isin(patients_to_remove)]
        # Remove these patients from labels
        filtered_labels = {pid: label for pid, label in labels.items() if pid not in patients_to_remove}
        # Convert and save the updated labels as a dataframe
        labels_df = pd.DataFrame({'patient_id': filtered_labels.keys(), 'label': filtered_labels.values()})
        print(f"Remaining patients after filtering: {labels_df.shape[0]}")
        print(f"Remaining patients after filtering: {combined_features_df['patient_id'].nunique()}")
        # make sure the value of birth date is in the correct format (timestamps) and convert the rest to numeric
        combined_features_df.loc[combined_features_df['feature_name'] == 'Birth Date', 'value'] = pd.to_datetime(
            combined_features_df.loc[combined_features_df['feature_name'] == 'Birth Date', 'value'], errors='coerce'
        )
        combined_features_df['time'] = pd.to_datetime(combined_features_df['time'])

        print(combined_features_df.isnull().sum())
        for index, row in combined_features_df.iterrows():
            if isinstance(row['value'], pd.Timestamp):
                continue  # Skip if it's already a timestamp
            try:
                combined_features_df.at[index, 'value'] = pd.to_numeric(row['value'], errors='coerce')
            except:
                continue  # Skip if conversion fails
        print(combined_features_df.isnull().sum())
        combined_features_df = combined_features_df.dropna() #drop the unneeded features
        print(combined_features_df.isnull().sum())
        combined_features_df = combined_features_df[~combined_features_df['feature_name'].isin(['Contact dermatitis', 'Application of moisturizer to skin','Skin condition care','Eczema'])]
        print(f"Data len after filtering:{len(combined_features_df)}")
        combined_features_df.to_csv(f'synthea_dataset_AtopicDermatitis_{output_num}.csv',index=False)
        labels_df.to_csv(f'labels_AtopicDermatitis_synthea_{output_num}.csv', index=False)        
        dfs = {
            "Birth Date": features_BD,
            "Allergies": allergies_features_df,
            "Careplans": careplans_features_df,
            "Medications": medications_features,
            "Procedures": procedures_features_df,
            "Observations": observations_features,
            "Immunizations": immunizations_features_df,
            "Encounters": encounters_features_df,
            "Conditions": conditions_features
        }

        for name, df in dfs.items():
            print(f"{name}: {df['feature_name'].nunique()}\n")
            
    else:
        # List of DataFrames to concatenate
        # dfs = [features_BD, allergies_features_df, careplans_features_df, medications_features,procedures_features_df,observations_features,immunizations_features_df,encounters_features_df,conditions_features]
        print(f'Reading combined final feature from output {output_num}..')
        combined_features_df = pd.read_csv(f'synthea_dataset_AtopicDermatitis_{output_num}.csv')
        labels_df = pd.read_csv(f'labels_AtopicDermatitis_synthea_{output_num}.csv')
        combined_features_df.loc[combined_features_df['feature_name'] == 'Birth Date', 'value'] = pd.to_datetime(combined_features_df.loc[combined_features_df['feature_name'] == 'Birth Date', 'value'])
        combined_features_df['time'] = pd.to_datetime(combined_features_df['time'])
        csv_files = [f for f in os.listdir(features_output_path) if f.endswith('.csv')]
        [careplans_features_df,medications_features,allergies_features_df,observations_features,procedures_features_df,features_BD,conditions_features,immunizations_features_df,encounters_features_df] = [pd.read_csv(os.path.join(features_output_path, file)) for file in csv_files]
        
        combined_features_df = combined_features_df[~combined_features_df['feature_name'].isin(['Contact dermatitis', 'Application of moisturizer to skin','Skin condition care','Eczema'])]
        print(f"Data len:{len(combined_features_df)}")
        

    return combined_features_df,labels_df,[careplans_features_df,medications_features,allergies_features_df,observations_features,procedures_features_df,features_BD,conditions_features,immunizations_features_df,encounters_features_df] 

#%% run:
create = False
output_num = 9
# Path to the dataset
# data_path = r"/mnt/drive2/datasets/Synthea/synthea_1m_fhir_3_0_May_24/output_9/csv/"
data_path = r"/mnt/drive2/datasets/Synthea/synthea_1m_fhir_3_0_May_24/output_{}/csv/".format(output_num)
features_output_path = '/home/noreena/Set-Tree/features_dfs_{}/'.format(output_num)
# List of relevant files
files = [
    "allergies.csv", "careplans.csv", "conditions.csv", "encounters.csv",
    "immunizations.csv", "medications.csv", "observations.csv", "patients.csv",
    "procedures.csv"
]

combined_features_df,labels_df,[careplans_features_df,medications_features,allergies_features_df,observations_features,procedures_features_df,features_BD,conditions_features,immunizations_features_df,encounters_features_df] = create_or_load_dataset(data_path, files,features_output_path,output_num,create=create)
# %%
first_file = os.path.join(data_path, files[0])
df = pd.read_csv(first_file, usecols=["PATIENT"])
common_patients = set(df["PATIENT"].unique())

# patients
patients = pd.read_excel('patients.xlsx')
patients=patients[patients['ID'].isin(common_patients)]
# Extract relevant columns
features_BD = patients[['ID', 'BIRTHDATE']].copy()
print(len(features_BD))
features_BD['BIRTHDATE'] = pd.to_datetime(features_BD['BIRTHDATE'], errors='coerce')

# Keep only rows where 'Birth Date' is a valid datetime (not NaT)
features_BD = features_BD.dropna(subset=['BIRTHDATE'])
print(len(features_BD))

# Rename columns to match the required format
features_BD['feature_name'] = "Birth Date"
features_BD['value'] = features_BD['BIRTHDATE']  # Date stored in 'value'
features_BD['time'] = features_BD['BIRTHDATE']   # Also stored in 'time'
# Rename patient ID column
features_BD.rename(columns={'ID': 'patient_id'}, inplace=True)
# Select final columns
features_BD = features_BD[['patient_id', 'time', 'feature_name', 'value']]
features_BD.to_csv(os.path.join(features_output_path, "BirthDate_features.csv.csv"),index=False)

# %%
procedures = pd.read_csv(os.path.join(data_path, "procedures.csv"))
procedures = procedures[procedures['PATIENT'].isin(common_patients)]
procedures_features = [
    "Fracture subluxation of wrist",
    "Rupture of patellar tendon",
    "Laceration of foot",
    "Facial laceration",
    "Fracture of ankle",
    "Fracture of clavicle",
    "Normal pregnancy",
    "Laceration of hand",
    "Laceration of thigh",
    "Closed fracture of hip",
    "Child attention deficit disorder",
    "Bullet wound",
    "Laceration of forearm",
    "Polyp of colon",
    "Recurrent rectal polyp",
    "Tubal pregnancy",
    "Third degree burn",
    "Fracture of forearm",
    "Fracture of rib",
    "Appendicitis",
    "Injury of medial collateral ligament of knee",
    "Injury of anterior cruciate ligament",
    "Injury of tendon of the rotator cuff of shoulder",
    "Secondary malignant neoplasm of colon",
    "Fracture of the vertebral column with spinal cord injury",
    "Primary malignant neoplasm of colon"
]

def create_feature_EHR(relevant_features,df,time_col_name='START',target_col='DESCRIPTION') :   
    # List to store the allergy feature data
    data = []
    # Loop through the common patients and check if they have any of the relevant allergies
    for patient in common_patients:
        for feature in relevant_features:
            # Filter the allergies data for the patient
            patient_data = df[df['PATIENT'] == patient]
            
            # Check if the patient has the allergy
            has_feature = 1 if any(patient_data[target_col] == feature) else 0
            
            # Get the 'START' column as time
            if has_feature == 1:
                time_start = patient_data[patient_data[target_col] == feature][time_col_name].min()
            else:
                time_start = None  # If no allergy, no start time

            # Add the allergy feature to the list
            data.append({
                'patient_id': patient,
                'time': time_start,
                'feature_name': feature,
                'value': has_feature
            })

    # Convert the list into a DataFrame
    features_df = pd.DataFrame(data)
    # Ensure that the DataFrame has the right format (repeated patient_id for each allergy)
    # Repeat for all patients and allergies
    features_df = features_df.dropna(subset=["time"])  # drop rows without valid time
    # Sort by patient_id and time
    features_df.sort_values(by=["patient_id", "time"], inplace=True)
    return features_df

procedures_features_df = create_feature_EHR(procedures_features,procedures,target_col='REASONDESCRIPTION',time_col_name='DATE') 
procedures_features_df.to_csv(os.path.join(features_output_path, "procedures_features.csv"),index=False)

# %%
#----------------------------------------
print('Combining features')
csv_files = [f for f in os.listdir(features_output_path) if f.endswith('.csv')]
[careplans_features,medications_features,allergies_features,observations_features,procedures_features,BirthDate_features,conditions_features,immunizations_features,encounters_features] = [pd.read_csv(os.path.join(features_output_path, file)) for file in csv_files]
dfs = [pd.read_csv(os.path.join(features_output_path, file)) for file in csv_files]   
# Concatenate all DataFrames vertically
combined_features_df = pd.concat(dfs, ignore_index=True)
# Convert labels dict keys (patient IDs) into a list in the given order
patient_order = list(labels.keys())
# Create a categorical column in the DataFrame to preserve the custom order
combined_features_df["patient_id"] = pd.Categorical(combined_features_df["patient_id"], categories=patient_order, ordered=True)
# Sort the DataFrame based on the custom patient order
combined_features_df = combined_features_df.sort_values(by=["patient_id", "time"])  
print('Filtering Data')  
#filter
# Get patients with NaN values in the 'value' column
patients_with_nan = combined_features_df[combined_features_df["value"].isna()]["patient_id"].unique()
print(f'remove {len(patients_with_nan)} patients because of nan.')

# Get patients that have the "Birth Date" feature
patients_with_birthdate = combined_features_df.loc[
    combined_features_df["feature_name"] == "Birth Date", "patient_id"
].unique()
# Find patients to remove (either have NaN values or lack Birth Date)
patients_to_remove = set(patients_with_nan) | set(labels.keys()) - set(patients_with_birthdate)
print(f'remove {len(patients_to_remove)} patients.')
# Remove these patients from combined_features_df
combined_features_df = combined_features_df[~combined_features_df["patient_id"].isin(patients_to_remove)]
# Remove these patients from labels
filtered_labels = {pid: label for pid, label in labels.items() if pid not in patients_to_remove}
# Convert and save the updated labels as a dataframe
labels_df = pd.DataFrame({'patient_id': filtered_labels.keys(), 'label': filtered_labels.values()})
print(f"Remaining patients after filtering: {labels_df.shape[0]}")
print(f"Remaining patients after filtering: {combined_features_df['patient_id'].nunique()}")
# make sure the value of birth date is in the correct format (timestamps) and convert the rest to numeric
combined_features_df.loc[combined_features_df['feature_name'] == 'Birth Date', 'value'] = pd.to_datetime(
    combined_features_df.loc[combined_features_df['feature_name'] == 'Birth Date', 'value'], errors='coerce'
)
print(combined_features_df.isnull().sum())
for index, row in combined_features_df.iterrows():
    if isinstance(row['value'], pd.Timestamp):
        continue  # Skip if it's already a timestamp
    try:
        combined_features_df.at[index, 'value'] = pd.to_numeric(row['value'], errors='coerce')
    except:
        continue  # Skip if conversion fails
print(combined_features_df.isnull().sum())
combined_features_df = combined_features_df.dropna() #drop the unneeded features
print(combined_features_df.isnull().sum())
combined_features_df = combined_features_df[~combined_features_df['feature_name'].isin(['Contact dermatitis', 'Application of moisturizer to skin','Skin condition care','Eczema'])]
print(f"Data len after filtering:{len(combined_features_df)}")
combined_features_df.to_csv('synthea_dataset_AtopicDermatitis_9.csv',index=False)
labels_df.to_csv('labels_AtopicDermatitis_synthea_9.csv', index=False)        

# %%
