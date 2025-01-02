import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import xarray as xr
import seaborn as sns

import pymc as pm 
import pymc_bart as pmb
import pytensor
import pytensor.tensor as pt

from pymc_bart.split_rules import ContinuousSplitRule, SubsetSplitRule
from scipy.special import logit
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
from sklearn.metrics import (
    mean_absolute_percentage_error
)
import datetime

pytensor.config.cxx = '/usr/bin/clang++'
# Plotting configuration
az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.facecolor"] = "white"

def load_transactions(filepath: str, sample_start_date: str = "2020-12-31") -> pd.DataFrame:
    """
    Loads dataframe and filters outliers (data points that occur too early in sample)

    Args:
        filepath (str): csv file containing transactions
        sample_start_date: orders before this date will be filtered out

    Returns:
        pd.DataFrame: dataframe with outliers filtered out
    """

    df = pd.read_csv(
        filepath,
    )

    try:
        assert("CUSTOMER_KEY" in df.columns)
        assert("ORDER_KEY" in df.columns)
        assert("DATE_KEY" in df.columns)
        df['CUSTOMER_KEY'] = df['CUSTOMER_KEY'].astype(str)
        df['ORDER_KEY'] = df['ORDER_KEY'].astype(str)
        df['DATE_KEY'] = pd.to_datetime(df['DATE_KEY'])
        mask = df['DATE_KEY'] > pd.to_datetime(sample_start_date)
        return df[mask]
    except AssertionError as e:
        print(f"Expected column not found, {e}")
        print(f"Returning None")
        return None


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Adds week, month, and cohort labels to transactions

    Args:
        df (pd.DataFrame): dataframe containing transactions

    Returns:
        pd.DataFrame
    """
    
    # week label
    df['WEEK'] = (
        df['DATE_KEY'] - 
        df['DATE_KEY'].dt.weekday * np.timedelta64(1, 'D')
    )

    #Month label
    df['MONTH'] = (
        df['DATE_KEY'] - 
        (df['DATE_KEY'].dt.day - 1) * np.timedelta64(1, 'D')
    )


    # cohort label
    first_transaction = df.groupby(['CUSTOMER_KEY'])['DATE_KEY'].min()
    first_transaction_y = first_transaction.dt.year
    first_transaction_m = first_transaction.dt.month
    cohort = pd.to_datetime(
        pd.DataFrame({
            "Year": first_transaction_y,
            "Month": first_transaction_m,
            "Day": np.ones(shape=first_transaction_m.shape)
        })
    )
    df = pd.merge(
        df,
        cohort.rename("COHORT"),
        left_on = 'CUSTOMER_KEY',
        right_index = True
    )

    return df

def preprocess_transactions_to_cohort(
        transactions: pd.DataFrame,
        project_out_of_sample: bool = True
    ) -> pd.DataFrame:
    """Pivot transactions dataframe into cohort dataframe

    Args:
        transactions (pd.DataFrame): 
            dataframe with one row per transaction, 
            identified by a CUSTOMER_KEY, ORDER_KEY, FULL_REVENUE,
            WEEK, MONTH, COHORT
        project_out_of_sample (bool): 
            if true, supports projecting revenue 1 year from latest transaction date
            else, only supports projecting revenue for in-sample dates
    Returns:
        pd.DataFrame: cohort-view of dataset, containing one row per
        cohort per period, 
    """

    cohort_pivot = pd.pivot_table(
        data = transactions,
        values = ["CUSTOMER_KEY", "FULL_REVENUE"],
        index = ["COHORT", "MONTH"],
        aggfunc={
            "CUSTOMER_KEY": lambda x: len(x.unique()),
            "FULL_REVENUE": "sum"
        },
        fill_value=0
    )

    cohort_pivot.reset_index(inplace=True)
    cohort_pivot.rename(columns={
        "COHORT": "cohort",
        "MONTH": "period",
        "CUSTOMER_KEY": "n_active_users",
        "FULL_REVENUE": "revenue"
        },
        inplace=True
    )

    cohort_count = transactions.groupby(['COHORT'])['CUSTOMER_KEY'].apply(
        lambda x: len(x.unique())
    )

    cohort_pivot = pd.merge(
        cohort_pivot, 
        cohort_count.rename("n_users"), 
        left_on="cohort", 
        right_index=True
    )

    max_date = transactions['DATE_KEY'].max()
    cohort_pivot['age'] = pd.to_datetime(max_date) - cohort_pivot['cohort']


    if project_out_of_sample:
        project_end_date = cohort_pivot['period'].max() + pd.DateOffset(years = 1)
        for cohort in cohort_pivot['cohort'].unique():
            mask = cohort_pivot['cohort'] == cohort 
            try:
                for month in pd.date_range(
                    start = cohort_pivot[mask]['period'].max() + pd.Timedelta(days = 31),
                    end = project_end_date,
                    freq="MS"
                ):
                    max_idx = cohort_pivot.index.max() + 1
                    cohort_pivot.loc[max_idx, "cohort"] = cohort
                    cohort_pivot.loc[max_idx, "period"] = month
                    cohort_pivot.loc[max_idx, "n_users"] = cohort_pivot[cohort_pivot['cohort'] == cohort]['n_users'].unique()[0]
                    cohort_pivot.loc[max_idx, "age"] = cohort_pivot[cohort_pivot['cohort'] == cohort]['age'].unique()[0]
            except ValueError as e:
                print(f"Failed to pad cohort {cohort} for month {month}, {e}")
        cohort_pivot.sort_values(['cohort', 'period'], inplace=True)

    cohort_pivot['cohort_age'] = cohort_pivot['period'] - cohort_pivot['cohort']
    cohort_pivot['retention'] = cohort_pivot['n_active_users'] / cohort_pivot['n_users']
    cohort_pivot['revenue_per_user'] = cohort_pivot['revenue'] / cohort_pivot['n_users']
    cohort_pivot['revenue_per_active_user'] = cohort_pivot['revenue'] / cohort_pivot['n_active_users']

    cohort_pivot['cohort_age'] /= pd.to_timedelta("1D")
    cohort_pivot['age'] /= pd.to_timedelta("1D")

    return cohort_pivot

def custom_train_test_split(
        cutoff: str, 
        input_df: pd.DataFrame,
        testing_period: str = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divides cohort-view dataframe into training and test set
    Dataframe is split by cutoff


    Args:
        cutoff (str): 
            date of prediction, e.g. "2024-01-01". 
            Data up to and including this day is used for model fitting
        input_df (pd.DataFrame): cohort-view of dataframe
        testing_period (str, optional): 
            duration of data to use for validation. 
            Defaults to None (all remaining data used for validation).
            Accepted values: ["1Y", "3M", "6M"]

    Raises:
        ValueError: if cutoff cannot be converted via np.datetime64()
        ValueError: if input_df does not contain columns "cohort" and "period"
        ValueError: if illegal argument passed to testing_period

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (training_data, testing_data)
    """
    try:
        np.datetime64(cutoff)
    except ValueError as e:
        raise ValueError(f"invalid cutoff, {e}")
    try:
        assert("cohort" in input_df.columns)
        assert("period" in input_df.columns)
    except AssertionError as e:
        raise ValueError(f"invalid input_df, {e}")
    try:
        assert(testing_period in ["1Y", "3M", "6M", None])
    except AssertionError as e:
        raise ValueError(f"invalid testing_period, {e}")

    if testing_period is None: 
        # no testing period specified, use all available data for testing
        period_train_test_split = cutoff
        train_data_df = input_df.query("period <= @period_train_test_split")
        test_data_df = input_df.query("period > @period_train_test_split")
    else: 
        period_train_test_split = cutoff
        if testing_period == "1M":
            period_test_end = (np.datetime64(cutoff) + pd.Timedelta(days = 1 * 31)).strftime(format="%Y-%m-%d")
        elif testing_period == "3M":
            period_test_end = (np.datetime64(cutoff) + pd.Timedelta(days = 1 * 31 * 3)).strftime(format="%Y-%m-%d")
        else:
            # 1Y testing period
            period_test_end = (np.datetime64(cutoff) + pd.Timedelta(days = 1 * 365)).strftime(format="%Y-%m-%d")

        train_data_df = input_df.query("period <= @period_train_test_split")
        test_data_df = input_df.query("period > @period_train_test_split & period <= @period_test_end")

    test_data_df = test_data_df[
        test_data_df["cohort"].isin(train_data_df["cohort"].unique())
    ]
    return (train_data_df, test_data_df)

def preprocess_train_test_data(
        input_df: pd.DataFrame,
        mode: str,
        limit_training_data: bool = None,
        seen_cohorts: pd.Series = None,
        cohort_encoder = None,
        age_scaler = None,
        cohort_age_scaler = None
    ) -> dict:
    """
    Preprocesses input dataframe into a dictionary where each element can be passed to 
    PYMC model

    Args:
        input_df (pd.DataFrame): train or test dataframe, obtained from the splitting of a cohort-view dataframe
        mode (str): "test" or "train", test mode expects cohorts and scalers
        seen_cohorts (pd.Series, optional): in-sample cohorts. Under testing mode, all other cohorts are filtered out of testing set. Defaults to None.
        cohort_encoder (_type_, optional): Defaults to None.
        age_scaler (_type_, optional): Defaults to None.
        cohort_age_scaler (_type_, optional): Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        dict: 
    """

    try:
        assert(mode in ["train", "test"])
    except AssertionError as e:
        raise ValueError("Illegal argument passed: {mode}. Must be 'train' or 'test'")
    
    try:
        if mode == "test":
            assert(seen_cohorts is not None)
            assert(cohort_encoder is not None)
            assert(age_scaler is not None)
            assert(cohort_age_scaler is not None)
    except AssertionError as e:
        raise ValueError("Missing argument, test mode specified but at least one required arguments is missing, {e}")

    eps = np.finfo(float).eps

    if limit_training_data and mode == "train":
        input_df = input_df[input_df['period'] > input_df['period'].max() - pd.DateOffset(years = 1)]

    # drop cohorts that have age 0
    data_red_df = input_df.query("cohort_age > 0").reset_index(drop = True)

    # filtering is now done in train_test_split
    if (mode == "test"):
        data_red_df = data_red_df[
            data_red_df['cohort'].isin(seen_cohorts)
        ].reset_index(drop=True)

    obs_idx = data_red_df.index.to_numpy()
    n_users = data_red_df['n_users'].to_numpy().astype(int)
    n_active_users = data_red_df['n_active_users'].to_numpy()
    retention = data_red_df['retention'].to_numpy()
    revenue = data_red_df['revenue'].to_numpy() + eps

    data_red_df['month'] = data_red_df['period'].dt.strftime("%m").astype(int)
    data_red_df['cohort_month'] = data_red_df['cohort'].dt.strftime("%m").astype(int)
    data_red_df['period_month'] = data_red_df['period'].dt.strftime("%m").astype(int)
    cohort = data_red_df['cohort'].to_numpy()
    period = data_red_df['period'].to_numpy()
    age = data_red_df['age'].to_numpy()
    cohort_age = data_red_df['cohort_age'].to_numpy()

    if mode == "train": 
        retention_logit = logit(retention + eps)
        revenue_per_user = revenue / (n_active_users + eps)

        cohort_encoder = LabelEncoder()
        cohort_idx = cohort_encoder.fit_transform(cohort).flatten()

        period_encoder = LabelEncoder()
        period_idx = period_encoder.fit_transform(period).flatten()

        age_scaler = MaxAbsScaler()
        age_scaled = age_scaler.fit_transform(age.reshape(-1, 1)).flatten()

        cohort_age_scaler = MaxAbsScaler()
        cohort_age_scaled = cohort_age_scaler.fit_transform(cohort_age.reshape(-1, 1)).flatten()
    
    if mode == "test":
        cohort_idx = cohort_encoder.transform(cohort).flatten()
        age_scaled = age_scaler.transform(age.reshape(-1, 1)).flatten()
        cohort_age_scaled = cohort_age_scaler.transform(cohort_age.reshape(-1, 1)).flatten()
    
    features: list[str] = ['age', 'cohort_age', 'month']
    x = data_red_df[features]

    return {
        "obs_idx": obs_idx,
        "n_users": n_users,
        "n_active_users": n_active_users,
        "retention": retention,
        "revenue": revenue,
        "features": features,
        "x" : x,
        "cohorts": None if mode == "test" else cohort,
        "cohort_idx": cohort_idx,
        "age_scaled": age_scaled,
        "cohort_age_scaled": cohort_age_scaled,
        "cohort_encoder": None if mode == "test" else cohort_encoder,
        "age_scaler": None if mode == "test" else age_scaler,
        "cohort_age_scaler": None if mode == "test" else cohort_age_scaler,
        "period_idx": None if mode == "test" else period_idx,
        "retention_logit": None if mode == "test" else retention_logit,
        "revenue_per_user": None if mode == "test" else revenue_per_user,
        "data_red_df": data_red_df
    }

def build_model(
        features: dict,
        use_default_priors: bool = True,
        informative_priors: dict = None
    ) -> pm.Model:
    try:
        expected_keys = [
            "obs_idx",
            "n_users",
            "n_active_users",
            "retention",
            "revenue",
            "features",
            "x",
            "cohorts",
            "cohort_idx",
            "age_scaled",
            "cohort_age_scaled",
            "period_idx",
            "retention_logit",
        ]
        assert(key in features for key in expected_keys)
    except AssertionError as e:
        raise ValueError(f"Expected key is missing: {e}")

    try:
        if not use_default_priors:
            assert informative_priors is not None
    except AssertionError as e:
        raise ValueError(f"Expected informative priors, {e}")

    model = pm.Model(coords = {"features": features["features"]})
    with model:
        model.add_coord(
            name="obs", 
            values=features["obs_idx"]
        )
        age_scaled = pm.Data(
            name="age_scaled", 
            value=features["age_scaled"], 
            dims="obs"
        )
        cohort_age_scaled = pm.Data(
            name="cohort_age_scaled", 
            value=features['cohort_age_scaled'], 
            dims="obs"
        )
        x = pm.Data(
            name="x", 
            value=features["x"], 
            dims=("obs", "feature")
        )
        n_users = pm.Data(
            name="n_users", 
            value=features["n_users"], 
            dims="obs"
        )
        n_active_users = pm.Data(
            name="n_active_users", 
            value=features["n_active_users"], 
            dims="obs"
        )
        revenue = pm.Data(
            name="revenue", 
            value=features["revenue"], 
            dims="obs"
        )

        # --- Priors ---
        if use_default_priors:
            intercept = pm.Normal(name="intercept", mu=np.array([1,]), sigma=np.array([1,]))
            b_age_scaled = pm.Normal(name="b_age_scaled", mu=0, sigma=1)
            b_cohort_age_scaled = pm.Normal(name="b_cohort_age_scaled", mu=0, sigma=1)
            b_age_cohort_age_interaction = pm.Normal(
                name="b_age_cohort_age_interaction", mu=0, sigma=1
            )
        else:
            intercept = pm.Normal(
                name="intercept",
                mu=float(informative_priors['intercept_posterior'].mean()),
                sigma=float(informative_priors['intercept_posterior'].std())
            )
            
            b_age_scaled = pm.Normal(
                name="b_age_scaled",
                mu=float(informative_priors['b_age_scaled_posterior'].mean()),
                sigma=float(informative_priors['b_age_scaled_posterior'].std())
            )
            
            b_cohort_age_scaled = pm.Normal(
                name="b_cohort_age_scaled",
                mu=float(informative_priors['b_cohort_age_posterior'].mean()),
                sigma=float(informative_priors['b_cohort_age_posterior'].std())
            )
            
            b_age_cohort_age_interaction = pm.Normal(
                name="b_age_cohort_age_interaction",
                mu=float(informative_priors['b_interaction_posterior'].mean()),
                sigma=float(informative_priors['b_interaction_posterior'].std())
            )

        # --- Parametrization ---
        # The BART component models the image of the retention rate under the
        # logit transform so that the range is not constrained to [0, 1].
        mu = pmb.BART(
            name="mu",
            X=x,
            Y=features["retention_logit"],
            m=50,
            response="mix",
            split_rules=[ContinuousSplitRule(), ContinuousSplitRule(), SubsetSplitRule()],
            dims="obs",
        )
        # We use the inverse logit transform to get the retention rate back into [0, 1].
        p = pm.Deterministic(name="p", var=pm.math.invlogit(mu), dims="obs")
        # We add a small epsilon to avoid numerical issues.
        eps = np.finfo(float).eps
        p = pt.switch(pt.eq(p, 0), eps, p)
        p = pt.switch(pt.eq(p, 1), 1 - eps, p)

        # For the revenue component we use a Gamma distribution where we combine the number
        # of estimated active users with the average revenue per user.
        lam_log = pm.Deterministic(
            name="lam_log",
            var=intercept
            + b_age_scaled * age_scaled
            + b_cohort_age_scaled * cohort_age_scaled
            + b_age_cohort_age_interaction * age_scaled * cohort_age_scaled,
            dims="obs",
        )

        lam = pm.Deterministic(name="lam", var=pm.math.exp(lam_log), dims="obs")

        n_active_users_estimated = pm.Binomial(
            name="n_active_users_estimated",
            n=n_users,
            p=p,
            observed=n_active_users,
            dims="obs",
        )

        x = pm.Gamma(
            name="revenue_estimated",
            alpha=n_active_users_estimated + eps,
            beta=lam,
            observed=revenue,
            dims="obs",
        )

        mean_revenue_per_active_user = pm.Deterministic(
            name="mean_revenue_per_active_user", var=(1 / lam), dims="obs"
        )
        pm.Deterministic(
            name="mean_revenue_per_user", var=p * mean_revenue_per_active_user, dims="obs"
        )
    
    return model

def fit_model(
        model: pm.model,
        draws:int = 1000,
        chains:int = 4,
        random_seed: int = 42
    ) -> tuple[
    pm.Model, 
    az.InferenceData, 
    (az.InferenceData | dict[str, np.ndarray])
    ]:
    with model:
        idata = pm.sample(draws = draws, chains = chains, random_seed=random_seed)
        posterior_predictive = pm.sample_posterior_predictive(trace = idata, random_seed=42)

    return (model, idata, posterior_predictive)

def extract_posteriors(idata: az.InferenceData) -> dict:
    posteriors = {}
    posteriors['intercept_posterior'] = idata.posterior['intercept']
    posteriors['b_age_scaled_posterior'] = idata.posterior['b_age_scaled']
    posteriors['b_cohort_age_posterior'] = idata.posterior['b_cohort_age_scaled']
    posteriors['b_interaction_posterior'] = idata.posterior['b_age_cohort_age_interaction']
    return posteriors


def draw_new_predictions(
        model: pm.model,
        test_features: dict,
        idata: az.InferenceData
    ) -> tuple[
    pm.Model, 
    az.InferenceData, 
    ]:
    
    try:
        expected_keys = [
            "obs_idx",
            "n_users",
            "n_active_users",
            "revenue",
            "x",
            "age_scaled",
            "cohort_age_scaled",
        ]
        assert(key in test_features for key in expected_keys)
    except AssertionError as e:
        raise ValueError(f"Expected key is missing: {e}")

    with model:
        pm.set_data(
            new_data={
                "age_scaled": test_features['age_scaled'],
                "cohort_age_scaled": test_features['cohort_age_scaled'],
                "x": test_features['x'],
                "n_users": test_features['n_users'],
                "n_active_users": np.ones_like(
                    test_features['n_active_users']
                ),  # Dummy data to make coords work! We are not using this at prediction time!
                "revenue": np.ones_like(
                    test_features['revenue']
                ),  # Dummy data to make coords work! We are not using this at prediction time!
            },
            coords={"obs": test_features['obs_idx']},
        )
        idata.extend(
            pm.sample_posterior_predictive(
                trace=idata,
                var_names=[
                    "p",
                    "mu",
                    "n_active_users_estimated",
                    "revenue_estimated",
                    "mean_revenue_per_user",
                    "mean_revenue_per_active_user",
                ],
                idata_kwargs={"coords": {"obs": test_features['obs_idx']}},
                random_seed=42,
            )
        )
        return (model, idata)
    

def process_idata_posterior_predictive_for_plotting(
    idata: az.data.inference_data.InferenceData,
    posterior_predictive: az.data.inference_data.InferenceData,
    train_data_red_df: pd.DataFrame,
    test_data_red_df: pd.DataFrame
)-> pd.DataFrame: 
    revenue_prediction = idata.posterior_predictive['revenue_estimated']
    mean_predicted_revenue = revenue_prediction.mean(dim = ['chain', 'draw'])
    hdi_predicted_revenue = az.hdi(revenue_prediction, hdi_prob = 0.95)['revenue_estimated']

    pos_revenue_prediction = posterior_predictive.posterior_predictive['revenue_estimated']
    pos_mean_predicted_revenue = pos_revenue_prediction.mean(dim = ['chain', 'draw'])
    pos_hdi_predicted_revenue = az.hdi(pos_revenue_prediction, hdi_prob = 0.95)['revenue_estimated']


    train_data_red_df['predicted_revenue'] = pos_mean_predicted_revenue
    train_data_red_df['hdi_lower'] = pos_hdi_predicted_revenue[:, 0]
    train_data_red_df['hdi_upper'] = pos_hdi_predicted_revenue[:, 1]

    pos_aggregate_predicted_revenue = train_data_red_df.groupby(['period'])['predicted_revenue'].sum()
    pos_aggregate_hdi_lower = train_data_red_df.groupby(['period'])['hdi_lower'].sum()
    pos_aggregate_hdi_upper = train_data_red_df.groupby(['period'])['hdi_upper'].sum()
    pos_aggregate_actual_revenue = train_data_red_df.groupby(['period'])['revenue'].sum()

    
    test_data_red_df['predicted_revenue'] = mean_predicted_revenue
    test_data_red_df['hdi_lower'] = hdi_predicted_revenue[:, 0]
    test_data_red_df['hdi_upper'] = hdi_predicted_revenue[:, 1]
    aggregate_predicted_revenue = test_data_red_df.groupby(['period'])['predicted_revenue'].sum()
    aggregate_hdi_lower = test_data_red_df.groupby(['period'])['hdi_lower'].sum()
    aggregate_hdi_upper = test_data_red_df.groupby(['period'])['hdi_upper'].sum()
    aggregate_actual_revenue = test_data_red_df.groupby(['period'])['revenue'].apply(
        lambda x: x.sum() if x.isna().sum() < x.shape[0] else np.nan
    )

    train_plot_data = pd.DataFrame({
        'Predicted Revenue': pos_aggregate_predicted_revenue,
        'Actual Revenue': pos_aggregate_actual_revenue,
        "hdi_lower": pos_aggregate_hdi_lower,
        "hdi_upper": pos_aggregate_hdi_upper,
    })
    plot_data = pd.DataFrame({
        'Predicted Revenue': aggregate_predicted_revenue,
        'Actual Revenue': aggregate_actual_revenue,
        "hdi_lower": aggregate_hdi_lower,
        "hdi_upper": aggregate_hdi_upper,
    })
    combined_data = pd.concat([train_plot_data, plot_data], keys=['Train', 'Test'])
    combined_data = combined_data.reset_index(level=0)
    combined_data = combined_data.rename(columns={'level_0': 'Dataset'})
    return combined_data

@ DeprecationWarning
def convert_to_forward_revenue(log_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    # log_df['period'] = pd.to_datetime(log_df['period'], format="mixed")
    # log_df['cutoff'] = pd.to_datetime(log_df['cutoff'], format="mixed")

    transformed_log_df_predicted = log_df.pivot_table(
        index = "cutoff",
        columns = "period",
        values = "Predicted Revenue"
    )

    transformed_log_df_predicted['3 month forward'] =  np.full(transformed_log_df_predicted.shape[0], np.nan)
    transformed_log_df_predicted['6 month forward'] =  np.full(transformed_log_df_predicted.shape[0], np.nan)
    transformed_log_df_predicted['12 month forward'] = np.full(transformed_log_df_predicted.shape[0], np.nan)

    num_cols = transformed_log_df_predicted.shape[1]

    for i, idx in enumerate(transformed_log_df_predicted.index):
        if idx < log_df['period'].max() - pd.Timedelta(days = 3 * 31):
            transformed_log_df_predicted.loc[idx, '3 month forward'] = np.sum(transformed_log_df_predicted.iloc[i, i:min(i+3, num_cols - 4)])
        if idx < log_df['period'].max() - pd.Timedelta(days = 6 * 31):
            transformed_log_df_predicted.loc[idx, '6 month forward'] = np.sum(transformed_log_df_predicted.iloc[i, i:min(i+6, num_cols - 4)])
        if idx < log_df['period'].max() - pd.Timedelta(days = 365):
            transformed_log_df_predicted.loc[idx, '12 month forward'] = np.sum(transformed_log_df_predicted.iloc[i,i:min(i+12, num_cols - 4)])


    transformed_log_df_actual = log_df.pivot_table(
        index = "cutoff",
        columns = "period",
        values = "Actual Revenue"
    )

    transformed_log_df_actual['3 month forward'] =  np.full( transformed_log_df_actual.shape[0], np.nan)
    transformed_log_df_actual['6 month forward'] =  np.full( transformed_log_df_actual.shape[0], np.nan)
    transformed_log_df_actual['12 month forward'] = np.full(transformed_log_df_actual.shape[0] , np.nan)

    for i, idx in enumerate(transformed_log_df_actual.index):
        if idx < log_df['cutoff'].max() - pd.Timedelta(days = 3 * 31):
            transformed_log_df_actual.loc[idx, '3 month forward'] =  np.sum(transformed_log_df_actual.iloc[i, i:min(i+3, num_cols-4)])
        if idx < log_df['cutoff'].max() - pd.Timedelta(days = 6 * 31):
            transformed_log_df_actual.loc[idx, '6 month forward'] =  np.sum(transformed_log_df_actual.iloc[i, i:min(i+6, num_cols - 4)])
        if idx < log_df['cutoff'].max() - pd.Timedelta(days = 365):
            transformed_log_df_actual.loc[idx, '12 month forward'] = np.sum(transformed_log_df_actual.iloc[i, i:min(i+12, num_cols - 4)])

    return (transformed_log_df_actual, transformed_log_df_predicted)

@ DeprecationWarning
def plot_forward_revenue(actual, predicted):

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(
        (predicted['3 month forward']),
        label = "3 month forward revenue - predicted",
        linestyle="--",
        color = "C0",
    )
    ax.plot(
        (predicted['6 month forward']),
        label = "6 month forward revenue - predicted",
        linestyle="--",
        color = "C1",
    )
    ax.plot(
        (predicted['12 month forward']),
        label = "12 month forward revenue - predicted",
        linestyle="--",
        color = "C2",
    )
    ax.plot(
        (actual['3 month forward']),
        label = "3 month forward revenue - realized",
        color = "C0",
        marker = "o"
    )
    ax.plot(
        (actual['6 month forward']),
        label = "6 month forward revenue - realized",
        color = "C1",
        marker = "o"
    )
    ax.plot(
        (actual['12 month forward']),
        label = "12 month forward revenue - realized",
        color = "C2",
        marker = "o"
    )
    fig.legend(loc = "center left", bbox_to_anchor = (1, 0.5))
    fig.suptitle("Forward revenue against time of projection")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
    plt.show()


def trim_sum(x: pd.Series):
    if x.isna().sum() > 0:
        return np.nan
    return x.sum()


def calculate_forward_revenue(
        log_df: pd.DataFrame, 
        months: list[int],
    ):
    max_period = log_df['period'].max()

    dfs = []

    for n_months in months: 
        results = []
        for cutoff_date, group in log_df.groupby('cutoff'):
            if cutoff_date + pd.DateOffset(months=n_months) > max_period:
                results.append({
                    'cutoff': cutoff_date,
                    f'{n_months} months projected': np.nan,
                    f'{n_months} months HDI Lower': np.nan,
                    f'{n_months} months HDI Upper': np.nan,
                    f'{n_months} months actual': np.nan
                })
                continue
                
            forward_n_months = group[
                (group['period'] > cutoff_date) & 
                (group['period'] <= cutoff_date + pd.DateOffset(months=n_months))
            ]
            
            total_projected = trim_sum(forward_n_months['Predicted Revenue'])
            total_hdi_lower = trim_sum(forward_n_months['hdi_lower'])
            total_hdi_upper = trim_sum(forward_n_months['hdi_upper'])
            total_actual = trim_sum(forward_n_months['Actual Revenue'])
            
            results.append({
                'cutoff': cutoff_date,
                f'{n_months} months projected': total_projected,
                f'{n_months} months actual': total_actual,
                f'{n_months} months HDI Lower': total_hdi_lower,
                f'{n_months} months HDI Upper': total_hdi_upper,
            })
        dfs.append(pd.DataFrame(results).set_index("cutoff"))
    
    res = pd.concat(dfs, axis = 1)
    
    return pd.DataFrame(res)

def evaluate_predictions(
        log_df: pd.DataFrame,
        testing_period = "1Y" 
):
    dfs = []
    for cutoff_date, group in log_df.groupby(['cutoff']):
        res_dict = {}
        group = group[~group['Actual Revenue'].isna()]
        end_period_mask = (group['period'] < cutoff_date[0] + pd.DateOffset(years=1))
        in_sample_mask = (group['Dataset'] == "Train") 

        res_dict['in_sample_MAPE'] = mean_absolute_percentage_error(
            y_true = group["Actual Revenue"][in_sample_mask],
            y_pred = group["Predicted Revenue"][in_sample_mask],
        )

        res_dict['out_of_sample_MAPE'] = mean_absolute_percentage_error(
            y_true =    group["Actual Revenue"][~in_sample_mask * end_period_mask],
            y_pred = group["Predicted Revenue"][~in_sample_mask * end_period_mask],
        )
        res_dict['aggregate_MAPE'] = mean_absolute_percentage_error(
            y_true =    group["Actual Revenue"][end_period_mask],
            y_pred = group["Predicted Revenue"][end_period_mask],
        )
        res_dict['in_sample_HDI_coverage'] = (
            (group["Actual Revenue"][in_sample_mask] >=  group["hdi_lower"][in_sample_mask]) &
            (group["Actual Revenue"][in_sample_mask] <=  group["hdi_upper"][in_sample_mask])
        ).sum() / np.sum(in_sample_mask)
        res_dict['out_of_sample_HDI_coverage'] = (
            (group["Actual Revenue"][~in_sample_mask * end_period_mask] >=  group["hdi_lower"][~in_sample_mask * end_period_mask]) &
            (group["Actual Revenue"][~in_sample_mask * end_period_mask] <=  group["hdi_upper"][~in_sample_mask * end_period_mask])
        ).sum() / np.sum(~in_sample_mask * end_period_mask)
        res_dict['aggregate_HDI_coverage'] = (
            (group["Actual Revenue"][end_period_mask] >=  group["hdi_lower"][end_period_mask]) &
            (group["Actual Revenue"][end_period_mask] <=  group["hdi_upper"][end_period_mask])
        ).sum() / np.sum(end_period_mask)
        res_dict['cutoff'] = cutoff_date
        dfs.append(pd.DataFrame(res_dict).set_index("cutoff"))

    res = pd.concat(dfs, axis=0)
    return res

def plot_monthly_revenue(
        combined_data: pd.DataFrame, 
        save_file_path: str = None, 
        ax: plt.Axes = None
):
    combined_data.set_index('period', inplace=True)
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 8))

        train_mask = combined_data['Dataset'] == 'Train'
        ax.fill_between(
            combined_data[train_mask].index, 
            combined_data[train_mask]['hdi_lower'], 
            combined_data[train_mask]['hdi_upper'], 
            alpha=0.3, 
            color='C0', 
            label='95% HDI (Train)'
        )
        ax.plot(
            combined_data[train_mask].index, 
            combined_data[train_mask]['Predicted Revenue'], 
            label='Mean Predicted Revenue (Train)',
            color="C0",
            # marker='o'
            linestyle = '--'
        )
        ax.plot(
            combined_data[train_mask].index, 
            combined_data[train_mask]['Actual Revenue'], 
            label='Actual Revenue (Train)',
            color="C0",
            marker='o',
        )

        test_mask = combined_data['Dataset'] == 'Test'
        ax.fill_between(
            combined_data[test_mask].index, 
            combined_data[test_mask]['hdi_lower'], 
            combined_data[test_mask]['hdi_upper'], 
            alpha=0.3, 
            color='C1', 
            label='95% HDI (Test)'
        )
        ax.plot(
            combined_data[test_mask].index, 
            combined_data[test_mask]['Predicted Revenue'], 
            label='Mean Predicted Revenue (Test)',
            color="C1",
            # marker='o'
            linestyle = '--'
        )
        ax.plot(
            combined_data[test_mask].index, 
            combined_data[test_mask]['Actual Revenue'], 
            label='Actual Revenue (Test)',
            color="C1",
            marker='o',
        )

        split_point = combined_data[train_mask].index[-1]
        ax.axvline(x=split_point, color='black', linestyle='--', label='Train/Test Split')


        ax.set_xlabel('Period')
        ax.set_ylabel('Revenue')
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        ax.set_title('Aggregate Predicted vs Actual Revenue (Train and Test)', fontsize=16)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

        plt.tight_layout()

        if save_file_path is not None:
            plt.savefig(save_file_path)
        plt.show()
    else:
        train_mask = combined_data['Dataset'] == 'Train'
        ax.fill_between(
            combined_data[train_mask].index, 
            combined_data[train_mask]['hdi_lower'], 
            combined_data[train_mask]['hdi_upper'], 
            alpha=0.3, 
            color='C0', 
            label='95% HDI (Train)'
        )
        ax.plot(
            combined_data[train_mask].index, 
            combined_data[train_mask]['Predicted Revenue'], 
            label='Mean Predicted Revenue (Train)',
            color="C0",
            # marker='o'
            linestyle = '--'
        )
        ax.plot(
            combined_data[train_mask].index, 
            combined_data[train_mask]['Actual Revenue'], 
            label='Actual Revenue (Train)',
            color="C0",
            marker='o',
        )

        test_mask = combined_data['Dataset'] == 'Test'
        ax.fill_between(
            combined_data[test_mask].index, 
            combined_data[test_mask]['hdi_lower'], 
            combined_data[test_mask]['hdi_upper'], 
            alpha=0.3, 
            color='C1', 
            label='95% HDI (Test)'
        )
        ax.plot(
            combined_data[test_mask].index, 
            combined_data[test_mask]['Predicted Revenue'], 
            label='Mean Predicted Revenue (Test)',
            color="C1",
            # marker='o'
            linestyle = '--'
        )
        ax.plot(
            combined_data[test_mask].index, 
            combined_data[test_mask]['Actual Revenue'], 
            label='Actual Revenue (Test)',
            color="C1",
            marker='o',
        )

        split_point = combined_data[train_mask].index[-1]
        ax.axvline(x=split_point, color='black', linestyle='--', label='Train/Test Split')


        ax.set_xlabel('Period')
        ax.set_ylabel('Revenue')
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        ax.set_title('Aggregate Predicted vs Actual Revenue (Train and Test)', fontsize=16)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

def plot_forward_revenue(
        forward_revenue: pd.DataFrame,
        show_CI: bool = False
):
    if show_CI:
        fig, axes = plt.subplots(3,1, figsize = (15, 15))
        axes[0].plot(
            forward_revenue['3 months projected'],
            label = "3 month forward revenue - predicted",
            linestyle="--",
            color = "C0",
        )
        axes[1].plot(
            forward_revenue['6 months projected'],
            label = "6 month forward revenue - predicted",
            linestyle="--",
            color = "C1",
        )

        axes[2].plot(
            forward_revenue['12 months projected'],
            label = "12 month forward revenue - predicted",
            linestyle="--",
            color = "C2",
        )
        axes[0].plot(
            forward_revenue['3 months actual'],
            label = "3 month forward revenue - realized",
            color = "C0",
            marker = 'o',
        )
        axes[1].plot(
            forward_revenue['6 months actual'],
            label = "6 month forward revenue - realized",
            color = "C1",
            marker = 'o',
        )
        axes[2].plot(
            forward_revenue['12 months actual'],
            label = "12 month forward revenue - realized",
            color = "C2",
            marker = 'o',
        )

        axes[0].fill_between(
            x = forward_revenue.index,
            y1 = forward_revenue['3 months HDI Lower'],
            y2 = forward_revenue['3 months HDI Upper'],
            color = "C0",
            alpha = .3
        )
        axes[1].fill_between(
            x = forward_revenue.index,
            y1 = forward_revenue['6 months HDI Lower'],
            y2 = forward_revenue['6 months HDI Upper'],
            color = "C1",
            alpha = .3
        )
        axes[2].fill_between(
            x = forward_revenue.index,
            y1 = forward_revenue['12 months HDI Lower'],
            y2 = forward_revenue['12 months HDI Upper'],
            color = "C2",
            alpha = .3
        )

        fig.legend(loc = "center left", bbox_to_anchor = (1, 0.5))
        fig.suptitle("Forward revenue against time of projection")
        axes[0].set_title("3 month forward")
        axes[1].set_title("6 month forward")
        axes[2].set_title("12 month forward")
        axes[0].yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        axes[1].yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        axes[2].yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(
            forward_revenue['3 months projected'],
            label = "3 month forward revenue - predicted",
            linestyle="--",
            color = "C0",
        )
        ax.plot(
            forward_revenue['6 months projected'],
            label = "6 month forward revenue - predicted",
            linestyle="--",
            color = "C1",
        )

        ax.plot(
            forward_revenue['12 months projected'],
            label = "12 month forward revenue - predicted",
            linestyle="--",
            color = "C2",
        )
        ax.plot(
            forward_revenue['3 months actual'],
            label = "3 month forward revenue - realized",
            color = "C0",
            marker = 'o'
        )
        ax.plot(
            forward_revenue['6 months actual'],
            label = "6 month forward revenue - realized",
            color = "C1",
            marker = 'o',
        )
        ax.plot(
            forward_revenue['12 months actual'],
            label = "12 month forward revenue - realized",
            color = "C2",
            marker = 'o',
        )


        fig.legend(loc = "center left", bbox_to_anchor = (1, 0.5))
        fig.suptitle("Forward revenue against time of projection")
        ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
        plt.show()

def plot_evaluations(
        evaluations: pd.DataFrame,
        plot_period_start: str = None,
        plot_period_end: str = None,
):

    fig, axes = plt.subplots(2, 1, figsize=(18, 8))
    if plot_period_start is None:
        plot_period_start = evaluations.index.min()
    if plot_period_end is None:
        plot_period_end = evaluations.index.max()

    mask = (evaluations.index >= plot_period_start) * (evaluations.index <= plot_period_end)

    axes[0].plot(
        evaluations['in_sample_HDI_coverage'][mask],
        label = "In-sample HDI_coverage",
        linestyle="-",
        marker = 'o',
        color="black"
    )
    axes[0].plot(
        evaluations['out_of_sample_HDI_coverage'][mask],
        label = "Out-of-sample HDI_coverage",
        linestyle="--",
        marker = '*',
        color="blue"
    )

    axes[0].plot(
        evaluations['aggregate_HDI_coverage'][mask],
        label = "Aggregate HDI_coverage",
        linestyle="-.",
        marker = 'P',
        color="black"
    )

    axes[1].plot(
        evaluations['in_sample_MAPE'][mask],
        label = "In-sample MAPE",
        linestyle="-",
        marker = 'o',
        color="black"
    )
    axes[1].plot(
        evaluations['out_of_sample_MAPE'][mask],
        label = "Out-of-sample MAPE",
        linestyle="--",
        marker = '*',
        color="blue"
    )

    axes[1].plot(
        evaluations['aggregate_MAPE'][mask],
        label = "Aggregate MAPE",
        linestyle="-.",
        marker = 'P',
        color="black"
    )

    axes[0].set_title("HDI_coverage against cutoff period")
    axes[1].set_title("MAPE against cutoff period")
    fig.legend(loc="center left", bbox_to_anchor = (1, .5))
    fig.suptitle("Model Evaluation")
    plt.tight_layout()
    plt.show()


    
if __name__ == "__main__":
    transactions = load_transactions('./transactions.csv')
    transactions = preprocess_transactions(transactions)
    cohort = preprocess_transactions_to_cohort(transactions)
    train_df, test_df = custom_train_test_split("2023-06-01", cohort)
    train_features = preprocess_train_test_data(train_df, mode = "train")

    model = build_model(train_features)
    model, idata, pos_predictive = fit_model(model)

    test_features = preprocess_train_test_data(
        test_df,
        mode="test",
        seen_cohorts = train_features['cohorts'],
        cohort_encoder = train_features['cohort_encoder'],
        age_scaler = train_features['age_scaler'],
        cohort_age_scaler= train_features['cohort_age_scaler']
    )
    mode, idata = draw_new_predictions(model, test_features, idata)

    combined_data = process_idata_posterior_predictive_for_plotting(
        idata = idata,
        posterior_predictive=pos_predictive,
        train_data_red_df=train_features['data_red_df'],
        test_data_red_df=test_features['data_red_df']
    )

    






    







