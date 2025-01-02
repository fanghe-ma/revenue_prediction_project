print(f"Importing")
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from arviz.labels import MapLabeller
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
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error
)
import datetime

pytensor.config.cxx = '/usr/bin/clang++'
# Plotting configuration
az.style.use("arviz-darkgrid")
plt.rcParams["figure.figsize"] = [12, 7]
plt.rcParams["figure.dpi"] = 100
plt.rcParams["figure.facecolor"] = "white"


sample_transactions = pd.read_csv(
    './transactions.csv',
    dtype={
        "CUSTOMER_KEY": str,
        "ORDER_KEY": str,
    }
)
sample_transactions['DATE_KEY'] = pd.to_datetime(sample_transactions['DATE_KEY'])
mask = sample_transactions['DATE_KEY'] > pd.to_datetime("2020-12-31")
sample_transactions = sample_transactions[mask]
sample_transactions.groupby(['DATE_KEY'])['ORDER_KEY'].count().head(20)


# Week label
sample_transactions['WEEK'] = (
    sample_transactions['DATE_KEY'] - 
    sample_transactions['DATE_KEY'].dt.weekday * np.timedelta64(1, 'D')
)

#Month label
sample_transactions['MONTH'] = (
    sample_transactions['DATE_KEY'] - 
    (sample_transactions['DATE_KEY'].dt.day - 1) * np.timedelta64(1, 'D')
)

first_transaction = sample_transactions.groupby(['CUSTOMER_KEY'])['DATE_KEY'].min()
first_transaction

first_transaction_y = first_transaction.dt.year
first_transaction_m = first_transaction.dt.month

cohort = pd.to_datetime(
    pd.DataFrame({
        "Year": first_transaction_y,
        "Month": first_transaction_m,
        "Day": np.ones(shape=first_transaction_m.shape)
    })
)

sample_transactions = pd.merge(
    sample_transactions,
    cohort.rename("COHORT"),
    left_on = 'CUSTOMER_KEY',
    right_index = True
)


cohort_pivot = pd.pivot_table(
    data = sample_transactions,
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


cohort_count = sample_transactions.groupby(['COHORT'])['CUSTOMER_KEY'].apply(lambda x: len(x.unique()))
cohort_pivot = pd.merge(cohort_pivot, cohort_count.rename("n_users"), left_on="cohort", right_index=True)


max_date = sample_transactions['DATE_KEY'].max()
cohort_pivot['age'] = pd.to_datetime(max_date) - cohort_pivot['cohort']
cohort_pivot['cohort_age'] = cohort_pivot['period'] - cohort_pivot['cohort']
cohort_pivot['retention'] = cohort_pivot['n_active_users'] / cohort_pivot['n_users']

df = cohort_pivot

df['revenue_per_user'] = df['revenue'] / df['n_users']
df['revenue_per_active_user'] = df['revenue'] / df['n_active_users']


df['cohort_age'] /= pd.to_timedelta("1D")
df['age'] /= pd.to_timedelta("1D")


def _custom_train_test_split(
        cutoff: str, 
        input_df: pd.DataFrame,
        testing_period: str = None,
        test_existing_cohorts_only: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            period_test_end = (np.datetime64(cutoff) + pd.Timedelta(days = 1 * 365)).strftime(format="%Y-%m-%d")

        train_data_df = input_df.query("period <= @period_train_test_split")
        test_data_df = input_df.query("period > @period_train_test_split & period <= @period_test_end")

    if test_existing_cohorts_only:
        test_data_df = test_data_df[
            test_data_df["cohort"].isin(train_data_df["cohort"].unique())
        ]
    else:
        pass
    return (train_data_df, test_data_df)


def _process_idata_posterior_predictive_for_plotting(
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
    aggregate_actual_revenue = test_data_red_df.groupby(['period'])['revenue'].sum()

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

def _plot_combined_data(combined_data: pd.DataFrame, save_file_path: str = None, ax: plt.Axes = None):
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



 
df_for_projection = df.copy()
df_for_projection['period'] = pd.to_datetime(df_for_projection['period'])

projection_end_date = df_for_projection['period'].max() + pd.Timedelta(days = 365)

for cohort in df_for_projection['cohort'].unique():
    mask = df_for_projection['cohort'] == cohort
    try:
        for month in pd.date_range(
            start = df_for_projection[mask]['period'].max() + pd.Timedelta(days = 31),
            end = projection_end_date,
            freq="MS"
        ):
            max_idx = df_for_projection.index.max() + 1
            df_for_projection.loc[max_idx, "cohort"] = cohort
            df_for_projection.loc[max_idx, "period"] = month
            df_for_projection.loc[max_idx, "n_users"] = df_for_projection[df_for_projection['cohort'] == cohort]['n_users'].unique()[0]
            df_for_projection.loc[max_idx, "age"] = df_for_projection[df_for_projection['cohort'] == cohort]['age'].unique()[0]
            df_for_projection.loc[max_idx, "cohort_age"] = (month - cohort) / pd.Timedelta(days = 1)
    except ValueError as e:
        print(cohort)
        print(mask)
df_for_projection.sort_values(['cohort', 'period'], inplace=True)

df_for_projection['n_users'] = df_for_projection['n_users'].astype(int)
df_for_projection



def evaluate_predictions(combined_data: pd.DataFrame) -> dict[str, float]: 
    res_dict = {}
    in_sample_mask = combined_data['Dataset'] == "Train"
    res_dict['in_sample_MAPE'] = mean_absolute_percentage_error(
        y_true = combined_data[in_sample_mask]["Actual Revenue"],
        y_pred = combined_data[in_sample_mask]["Predicted Revenue"],
    )
    res_dict['out_of_sample_MAPE'] = mean_absolute_percentage_error(
        y_true = combined_data[~in_sample_mask]["Actual Revenue"],
        y_pred = combined_data[~in_sample_mask]["Predicted Revenue"],
    )
    res_dict['aggregate_MAPE'] = mean_absolute_percentage_error(
        y_true = combined_data["Actual Revenue"],
        y_pred = combined_data["Predicted Revenue"],
    )
    res_dict['in_sample_HDI_coverage'] = (
        (combined_data[in_sample_mask]["Actual Revenue"] >=  combined_data[in_sample_mask]["hdi_lower"]) &
        (combined_data[in_sample_mask]["Actual Revenue"] <=  combined_data[in_sample_mask]["hdi_upper"])
    ).sum() / np.sum(in_sample_mask)
    res_dict['out_of_sample_HDI_coverage'] = (
        (combined_data[~in_sample_mask]["Actual Revenue"] >=  combined_data[~in_sample_mask]["hdi_lower"]) &
        (combined_data[~in_sample_mask]["Actual Revenue"] <=  combined_data[~in_sample_mask]["hdi_upper"])
    ).sum() / np.sum(~in_sample_mask)
    res_dict['aggregate_HDI_coverage'] = (
        (combined_data["Actual Revenue"] >=  combined_data["hdi_lower"]) &
        (combined_data["Actual Revenue"] <=  combined_data["hdi_upper"])
    ).sum() / combined_data.shape[0]

    return res_dict



def _run_iteration_with_loss_function_extended_projections(
        cutoff: str, 
        df: pd.DataFrame, 
        log_file_path: str,
        loss_file_path: str,
        ax: plt.Axes = None
):
    print(f"Processing {cutoff}")
    print(f"Building model...")
    train_data_df, test_data_df = _custom_train_test_split(cutoff, df, testing_period = "1Y")

    eps = np.finfo(float).eps
    train_data_red_df = train_data_df.query("cohort_age > 0").reset_index(drop=True)
    train_obs_idx = train_data_red_df.index.to_numpy()
    train_n_users = train_data_red_df["n_users"].to_numpy().astype(int)
    train_n_active_users = train_data_red_df["n_active_users"].to_numpy()
    train_retention = train_data_red_df["retention"].to_numpy()
    train_retention_logit = logit(train_retention + eps)
    train_data_red_df["month"] = train_data_red_df["period"].dt.strftime("%m").astype(int)
    train_data_red_df["cohort_month"] = (
        train_data_red_df["cohort"].dt.strftime("%m").astype(int)
    )
    train_data_red_df["period_month"] = (
        train_data_red_df["period"].dt.strftime("%m").astype(int)
    )
    train_revenue = train_data_red_df["revenue"].to_numpy() + eps
    train_revenue_per_user = train_revenue / (train_n_active_users + eps)

    train_cohort = train_data_red_df["cohort"].to_numpy()
    train_cohort_encoder = LabelEncoder()
    train_cohort_idx = train_cohort_encoder.fit_transform(train_cohort).flatten()
    train_period = train_data_red_df["period"].to_numpy()
    train_period_encoder = LabelEncoder()
    train_period_idx = train_period_encoder.fit_transform(train_period).flatten()

    features: list[str] = ["age", "cohort_age", "month"]
    x_train = train_data_red_df[features]

    train_age = train_data_red_df["age"].to_numpy()
    train_age_scaler = MaxAbsScaler()
    train_age_scaled = train_age_scaler.fit_transform(train_age.reshape(-1, 1)).flatten()
    train_cohort_age = train_data_red_df["cohort_age"].to_numpy()
    train_cohort_age_scaler = MaxAbsScaler()
    train_cohort_age_scaled = train_cohort_age_scaler.fit_transform(
        train_cohort_age.reshape(-1, 1)
    ).flatten()


    test_data_red_df = test_data_df.query("cohort_age > 0")
    test_data_red_df = test_data_red_df[
        test_data_red_df["cohort"].isin(train_data_red_df["cohort"].unique())
    ].reset_index(drop=True)
    test_obs_idx = test_data_red_df.index.to_numpy()
    test_n_users = test_data_red_df["n_users"].to_numpy().astype(int)
    test_n_active_users = test_data_red_df["n_active_users"].to_numpy() + eps
    test_retention = test_data_red_df["retention"].to_numpy() 
    test_revenue = test_data_red_df["revenue"].to_numpy()

    test_cohort = test_data_red_df["cohort"].to_numpy()
    test_cohort_idx = train_cohort_encoder.transform(test_cohort).flatten()

    test_data_red_df["month"] = test_data_red_df["period"].dt.strftime("%m").astype(int)
    test_data_red_df["cohort_month"] = (
        test_data_red_df["cohort"].dt.strftime("%m").astype(int)
    )
    test_data_red_df["period_month"] = (
        test_data_red_df["period"].dt.strftime("%m").astype(int)
    )
    x_test = test_data_red_df[features]

    test_age = test_data_red_df["age"].to_numpy()
    test_age_scaled = train_age_scaler.transform(test_age.reshape(-1, 1)).flatten()
    test_cohort_age = test_data_red_df["cohort_age"].to_numpy()
    test_cohort_age_scaled = train_cohort_age_scaler.transform(
        test_cohort_age.reshape(-1, 1)
    ).flatten()

    with pm.Model(coords={"feature": features}) as model:
        # --- Data ---
        # model.add_coord(name="obs", values=train_obs_idx, mutable=True)
        # age_scaled = pm.MutableData(name="age_scaled", value=train_age_scaled, dims="obs")
        # cohort_age_scaled = pm.MutableData(
        #     name="cohort_age_scaled", value=train_cohort_age_scaled, dims="obs"
        # )
        # x = pm.MutableData(name="x", value=x_train, dims=("obs", "feature"))
        # n_users = pm.MutableData(name="n_users", value=train_n_users, dims="obs")
        # n_active_users = pm.MutableData(
        #     name="n_active_users", value=train_n_active_users, dims="obs"
        # )
        # revenue = pm.MutableData(name="revenue", value=train_revenue, dims="obs")

        model.add_coord(name="obs", values=train_obs_idx)
        age_scaled = pm.Data(name="age_scaled", value=train_age_scaled, dims="obs")
        cohort_age_scaled = pm.Data(
            name="cohort_age_scaled", value=train_cohort_age_scaled, dims="obs"
        )
        x = pm.Data(name="x", value=x_train, dims=("obs", "feature"))
        n_users = pm.Data(name="n_users", value=train_n_users, dims="obs")
        n_active_users = pm.Data(
            name="n_active_users", value=train_n_active_users, dims="obs"
        )
        revenue = pm.Data(name="revenue", value=train_revenue, dims="obs")

        # --- Priors ---
        intercept = pm.Normal(name="intercept", mu=np.array([1,]), sigma=np.array([1,]))
        b_age_scaled = pm.Normal(name="b_age_scaled", mu=0, sigma=1)
        b_cohort_age_scaled = pm.Normal(name="b_cohort_age_scaled", mu=0, sigma=1)
        b_age_cohort_age_interaction = pm.Normal(
            name="b_age_cohort_age_interaction", mu=0, sigma=1
        )

        # --- Parametrization ---
        # The BART component models the image of the retention rate under the
        # logit transform so that the range is not constrained to [0, 1].
        mu = pmb.BART(
            name="mu",
            X=x,
            Y=train_retention_logit,
            m=50,
            # response="mix",
            split_rules=[ContinuousSplitRule(), ContinuousSplitRule(), SubsetSplitRule()],
            dims="obs",
        )
        # We use the inverse logit transform to get the retention rate back into [0, 1].
        p = pm.Deterministic(name="p", var=pm.math.invlogit(mu), dims="obs")
        # We add a small epsilon to avoid numerical issues.
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

    with model:
        print(f"Fitting model")
        idata = pm.sample(draws=1000, chains = 4, random_seed=42)
        posterior_predictive = pm.sample_posterior_predictive(trace=idata, random_seed=42)

        print(f"Drawing predictions")
        pm.set_data(
            new_data={
                "age_scaled": test_age_scaled,
                "cohort_age_scaled": test_cohort_age_scaled,
                "x": x_test,
                "n_users": test_n_users,
                "n_active_users": np.ones_like(
                    test_n_active_users
                ),  # Dummy data to make coords work! We are not using this at prediction time!
                "revenue": np.ones_like(
                    test_revenue
                ),  # Dummy data to make coords work! We are not using this at prediction time!
            },
            coords={"obs": test_obs_idx},
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
                idata_kwargs={"coords": {"obs": test_obs_idx}},
                random_seed=42,
            )
        )
    
    print(f"Saving files to project directory")
    idata.to_json(filename = f"./idata_{cutoff}.json")
    posterior_predictive.to_json(filename = f"./posterior_predictive_{cutoff}.json")

    print(f"Generating visualization")
    combined_data = _process_idata_posterior_predictive_for_plotting(
        idata = idata,
        posterior_predictive=posterior_predictive,
        train_data_red_df=train_data_red_df,
        test_data_red_df=test_data_red_df,
    )
    combined_data['cutoff'] = [cutoff for _ in range(combined_data.shape[0])]
    combined_data.reset_index(inplace=True)

    log_df = pd.read_csv(log_file_path, index_col=0)
    log_df = pd.concat([log_df, combined_data], ignore_index=True)
    log_df.to_csv(log_file_path)

    loss_df = pd.read_csv(loss_file_path, index_col=0)
    loss_dict = evaluate_predictions(combined_data)
    loss_dict['train_test_split'] = cutoff

    loss_df = pd.concat(
        [loss_df, pd.DataFrame(
            [loss_dict]
        )],
        ignore_index=True
    )
    loss_df.to_csv(loss_file_path)

    if ax is not None:
        _plot_combined_data(
            combined_data=combined_data,
            # save_file_path=f'./{cutoff}.png',
            ax = ax,
        )

cutoff = "2022-01-01"
print(f"Testing with {cutoff}")
loss_file_path = './model_loss_1226.csv'
log_file_path = './model_log_1226.csv'
combined_data = _run_iteration_with_loss_function_extended_projections(
        cutoff=cutoff, 
        df = df_for_projection, 
        loss_file_path = loss_file_path,
        log_file_path = log_file_path
    )