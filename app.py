import pickle
import streamlit as st
import csv
import pandas as pd

data_file = './data/aps_failure_test_set.csv'
model_file = './final_model.pickle'
threshold = 0.05


@st.cache
def load_data(filepath):
    data = pd.read_csv(filepath, skiprows=20, na_values='na', nrows=50)
    dropped_columns = ['class', 'ab_000', 'ad_000', 'bk_000', 'bl_000', 'bm_000', 'bn_000', 'bo_000',
                       'bp_000', 'bq_000', 'br_000', 'cf_000', 'cg_000', 'ch_000', 'co_000',
                       'cr_000', 'ct_000', 'cu_000', 'cv_000', 'cx_000', 'cy_000', 'cz_000',
                       'da_000', 'db_000', 'dc_000']
    data.drop(columns=dropped_columns, inplace=True)
    return data


def load_model(filepath):
    with open(filepath, 'rb') as pickle_in:
        model = pickle.load(pickle_in)
    return model


def prediction_input(df, row_nr):
    input_dict = {}
    for i, header in enumerate(df.columns):
        input_dict[header] = st.sidebar.number_input(header, value=df.iloc[row_nr, i])
    input_df = pd.DataFrame(input_dict, index=[0])
    return input_df


def get_prediction(input_df, model, threshold):
    prob_score = model.predict_proba(input_df)[:, 1]
    return prob_score > threshold


def main(data_file=data_file, model_file=model_file):
    st.title("Is failure likely related to the APS?")
    st.sidebar.title('User Input')
    data = load_data(data_file)
    st.subheader(f'{data.shape[0]} entries loaded, pick one by entering a number between 0 and {data.shape[0] - 1}')
    row_nr = st.number_input("select input row", value=1, min_value=0, max_value=data.shape[0])
    pred_input = prediction_input(data, row_nr)
    model = load_model(model_file)
    if st.button("Predict"):
        result = get_prediction(pred_input, model, threshold)
        if result:
            st.write("This failure is likely related to the APS")
        else:
            st.write("This is failure is likely related to another system")


if __name__ == "__main__":
    main()
