### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME OVERVIEW --------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### PACKAGES ###
from Functions import *
from pathlib import Path
import os
DATA_DIR = os.getenv('DATA_DIR', 'data')

### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME OVERVIEW --------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

### EQUITIES ###
with open(Path(DATA_DIR) / 'SPX.csv', 'rb') as file:
    sp500 = pd.read_csv(file)
sp500.index = pd.to_datetime(sp500['Date']).values
sp500.drop('Date', axis=1, inplace=True)
spx_monthly = pd.DataFrame(sp500['Close']).resample('ME').last()
spx_monthly.columns = ['spx']

### BONDS ###
with open(Path(DATA_DIR) / 'AGG.csv', 'rb') as file:
    agg = pd.read_csv(file)
agg.index = pd.to_datetime(agg['Date']).values
agg.drop('Date', axis=1, inplace=True)
bonds_monthly = pd.DataFrame(agg['Close']).resample('ME').last()
bonds_monthly.columns = ['bonds']

### BCOM ###
with open(Path(DATA_DIR) / '^BCOM.csv', 'rb') as file:
    bcom = pd.read_csv(file)
bcom.index = pd.to_datetime(bcom['Date']).values
bcom.drop('Date', axis=1, inplace=True)
bcom_monthly = pd.DataFrame(bcom['Close']).resample('ME').last()
bcom_monthly.columns = ['bcom']

### YIELDS ###
with open(Path(DATA_DIR) / 'treasury_1m.pkl', 'rb') as file:
    treasury_1m = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'treasury_2y.pkl', 'rb') as file:
    treasury_2y = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'treasury_5y.pkl', 'rb') as file:
    treasury_5y = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'treasury_10y.pkl', 'rb') as file:
    treasury_10y = pd.read_pickle(file)
with open(Path(DATA_DIR) / 'treasury_30y.pkl', 'rb') as file:
    treasury_30y = pd.read_pickle(file)

treasury_merge = merge_dfs([treasury_1m,treasury_2y, treasury_5y, treasury_10y,treasury_30y]).dropna()
treasury_merge.index = pd.to_datetime(treasury_merge.index).values
treasury_merge.columns = ['1m','2y','5y','10y','30y']
treasury_monthly_df = treasury_merge.resample('ME').last()


### LIQUIDITY ###
with open(Path(DATA_DIR) / 'treasury.pkl', 'rb') as file:
    treasury = pd.read_pickle(file)


### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME OVERVIEW --------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_grid_nowcast():
    def build_prometheus_feature_matrices():
        # GROWTH TARGET
        growth = pdr.DataReader(
            'PCEC96',
            'fred',
            '1949-12-31',
            '2025-12-31'
        )
        growth.index = pd.to_datetime(growth.index) + pd.DateOffset(months=3)
        growth = growth.resample('ME').last()

        # INFLATION TARGET
        inflation = pdr.DataReader(
            'CPIAUCSL',
            'fred',
            '1949-12-31',
            '2025-12-31'
        )
        inflation.index = pd.to_datetime(inflation.index) + pd.DateOffset(months=2)
        inflation = inflation.resample('ME').last()

        # Load prebuilt lagged feature panels
        with open(Path(DATA_DIR) / 'growth_lagged_features.pkl', 'rb') as file:
            growth_lagged_features = pd.read_pickle(file)

        with open(Path(DATA_DIR) / 'inflation_lagged_features.pkl', 'rb') as file:
            inflation_lagged_features = pd.read_pickle(file)

        # GROWTH DATA
        growth_roc_2 = growth.pct_change(12).diff(3)
        growth_roc_2.columns = ['growth_roc_2']
        lagged_growth_roc_2 = growth_roc_2.shift(-1)
        lagged_growth_roc_2.columns = ['growth_roc_2_lag']

        # INFLATION DATA
        inflation_roc_2 = inflation.pct_change(12).diff(3)
        inflation_roc_2.columns = ['inflation_roc_2']
        lagged_inflation_roc_2 = inflation_roc_2.shift(-1)
        lagged_inflation_roc_2.columns = ['inflation_roc_2_lag']

        # MERGE TARGETS AND FEATURES (same as backtest)
        target_feature_merge = (
            merge_dfs([
                growth_lagged_features.diff(12).diff(3),
                inflation_lagged_features.diff(12).diff(3),
                lagged_growth_roc_2,
                lagged_inflation_roc_2,
            ]).dropna()
        )

        growth_vars = growth_lagged_features.columns
        inflation_vars = inflation_lagged_features.columns

        growth_features_targets = target_feature_merge[growth_vars].copy()
        growth_features_targets['growth_roc_2_lag'] = target_feature_merge['growth_roc_2_lag']

        inflation_features_targets = target_feature_merge[inflation_vars].copy()
        inflation_features_targets['inflation_roc_2_lag'] = target_feature_merge['inflation_roc_2_lag']

        growth_features_targets = growth_features_targets.dropna()
        inflation_features_targets = inflation_features_targets.dropna()

        return growth_features_targets, inflation_features_targets

    # --------------------------------------------------------------------------------
    # Core: one-step-ahead nowcast for a generic feature/target matrix
    #        using your PCA + linear + logistic structure
    # --------------------------------------------------------------------------------

    def prometheus_nowcast_single_block(
            feature_target_df,
            lookback_window_reg,
            pca_num_components_reg,
            lookback_window_cls,
            pca_num_components_cls):
        """
        Builds full rolling history (as in your backtest) and then
        uses *all* data up to the last available row to nowcast the
        sign of the next period's target.

        Returns:
            {
              'last_date': last in-sample date,
              'reg_point_forecast': float (continuous),
              'reg_sign': +1/-1,
              'cls_sign': +1/-1,
              'cls_proba_pos': float in [0,1],
              'combined_sign': +1/-1/0
            }
        """

        feature_cols = feature_target_df.columns[:-1]
        target_col = feature_target_df.columns[-1]

        # ----------------- Build rolling regression history (same logic) -----------------
        reg_preds = []
        reg_actuals = []
        reg_dates = []

        for row in range(lookback_window_reg + 1, len(feature_target_df) + 1):
            window_subset = feature_target_df[row - lookback_window_reg - 1:row]
            target_window_subset = window_subset[:lookback_window_reg]
            current_subset = window_subset[lookback_window_reg:]

            X_window = target_window_subset.loc[:, feature_cols]
            y_window = target_window_subset.loc[:, target_col]
            X_current = current_subset.loc[:, feature_cols]
            y_current = current_subset.loc[:, target_col]

            scaler = StandardScaler()
            X_window_scaled = scaler.fit_transform(X_window)
            X_current_scaled = scaler.transform(X_current)

            pca = PCA(n_components=pca_num_components_reg)
            factors_window = pca.fit_transform(X_window_scaled)
            factors_current = pca.transform(X_current_scaled)

            model = LinearRegression()
            model.fit(factors_window, y_window)
            current_pred = model.predict(factors_current)

            reg_preds.append(current_pred[0])
            reg_actuals.append(y_current.iloc[0])
            reg_dates.append(feature_target_df.index[row - 1])

        df_reg = pd.DataFrame(
            {'prediction': reg_preds, 'actual': reg_actuals},
            index=reg_dates
        )

        # ----------------- Build rolling logistic history (same logic) ------------------
        cls_pred_sign = []
        cls_proba_pos = []
        cls_actual_sign = []
        cls_dates = []

        for row in range(lookback_window_cls + 1, len(feature_target_df) + 1):
            window_subset = feature_target_df[row - lookback_window_cls - 1:row]
            target_window_subset = window_subset[:lookback_window_cls]
            current_subset = window_subset[lookback_window_cls:]

            X_window = target_window_subset.loc[:, feature_cols]
            y_window = target_window_subset.loc[:, target_col]
            X_current = current_subset.loc[:, feature_cols]
            y_current = current_subset.loc[:, target_col]

            y_current_val = y_current.iloc[0]
            actual_sign = 1 if y_current_val > 0 else -1

            y_window_class = (y_window > 0).astype(int)

            scaler = StandardScaler()
            X_window_scaled = scaler.fit_transform(X_window)
            X_current_scaled = scaler.transform(X_current)

            pca = PCA(n_components=pca_num_components_cls)
            factors_window = pca.fit_transform(X_window_scaled)
            factors_current = pca.transform(X_current_scaled)

            if y_window_class.nunique() >= 2:
                clf = LogisticRegression()
                clf.fit(factors_window, y_window_class)
                proba = clf.predict_proba(factors_current)[0]
                proba_pos = proba[1]
                pred_class = clf.predict(factors_current)[0]
            else:
                only_class = int(y_window_class.iloc[0])
                pred_class = only_class
                proba_pos = 1.0 if only_class == 1 else 0.0

            pred_sign = 1 if pred_class == 1 else -1

            cls_pred_sign.append(pred_sign)
            cls_proba_pos.append(proba_pos)
            cls_actual_sign.append(actual_sign)
            cls_dates.append(feature_target_df.index[row - 1])

        df_cls = pd.DataFrame(
            {
                'pred_sign': cls_pred_sign,
                'proba_pos': cls_proba_pos,
                'actual_sign': cls_actual_sign
            },
            index=cls_dates
        )

        # ----------------- Nowcast for *next* period: refit on full sample -----------------
        # Use the same windows but now train on all history and predict for last row

        # Last in-sample date is the last index of feature_target_df
        last_idx = feature_target_df.index[-1]
        X_full = feature_target_df.loc[:, feature_cols]
        y_full = feature_target_df.loc[:, target_col]

        # Regression part
        scaler_reg = StandardScaler()
        X_full_scaled = scaler_reg.fit_transform(X_full)
        pca_reg = PCA(n_components=min(pca_num_components_reg, X_full_scaled.shape[1]))
        factors_full = pca_reg.fit_transform(X_full_scaled)
        reg_model = LinearRegression()
        reg_model.fit(factors_full, y_full)

        # We treat the last row as "current features" and produce a one-step-ahead nowcast.
        # Strictly speaking, you would want features for t to predict y_{t+1}. Your original
        # labeling already shifts the target, so the last row corresponds to the last y.
        # In live use, you would feed in the *new* row; here we just use the last row pattern.

        X_last = X_full.iloc[[-1], :]
        X_last_scaled = scaler_reg.transform(X_last)
        factors_last = pca_reg.transform(X_last_scaled)
        reg_point_forecast = float(reg_model.predict(factors_last)[0])
        reg_sign = 1 if reg_point_forecast > 0 else -1

        # Classifier part
        scaler_cls = StandardScaler()
        X_full_scaled_cls = scaler_cls.fit_transform(X_full)
        pca_cls = PCA(n_components=min(pca_num_components_cls, X_full_scaled_cls.shape[1]))
        factors_full_cls = pca_cls.fit_transform(X_full_scaled_cls)
        y_full_class = (y_full > 0).astype(int)

        if y_full_class.nunique() >= 2:
            clf_full = LogisticRegression()
            clf_full.fit(factors_full_cls, y_full_class)
            X_last_scaled_cls = scaler_cls.transform(X_last)
            factors_last_cls = pca_cls.transform(X_last_scaled_cls)
            proba_last = clf_full.predict_proba(factors_last_cls)[0]
            proba_pos_last = float(proba_last[1])
            cls_sign = 1 if clf_full.predict(factors_last_cls)[0] == 1 else -1
        else:
            only_class = int(y_full_class.iloc[0])
            cls_sign = 1 if only_class == 1 else -1
            proba_pos_last = 1.0 if only_class == 1 else 0.0

        # Simple ensemble rule: if both agree, use that sign; otherwise neutral (0)
        if reg_sign == cls_sign:
            combined_sign = reg_sign
        else:
            combined_sign = 0

        return {
            'last_date': last_idx,
            'reg_point_forecast': reg_point_forecast,
            'reg_sign': reg_sign,
            'cls_sign': cls_sign,
            'cls_proba_pos': proba_pos_last,
            'combined_sign': combined_sign
        }

    # --------------------------------------------------------------------------------
    # Regime mapping + top-level function
    # --------------------------------------------------------------------------------

    def map_to_regime(growth_sign, inflation_sign):
        """
        growth_sign: +1 for improving growth (growth_roc_2_lag > 0), -1 for deteriorating
        inflation_sign: +1 for rising inflation, -1 for falling
        """
        if growth_sign > 0 and inflation_sign < 0:
            return 'Goldilocks'  # stronger growth, falling inflation
        elif growth_sign > 0 and inflation_sign > 0:
            return 'Reflation'  # stronger growth, rising inflation
        elif growth_sign < 0 and inflation_sign > 0:
            return 'Stagflation'  # weaker growth, rising inflation
        elif growth_sign < 0 and inflation_sign < 0:
            return 'Deflation'  # weaker growth, falling inflation
        else:
            return 'Unknown'

    def prometheus_regime_nowcast():
        """
        Master function to:
          - rebuild growth & inflation feature matrices
          - estimate next-period growth & inflation signs
          - map to a 4-quadrant regime

        Returns a dict:
          {
            'as_of_date': ...,
            'growth': {
                'point_forecast': ...,
                'sign': +1/-1/0,
                'cls_sign': ...,
                'cls_proba_pos': ...
            },
            'inflation': {
                same keys...
            },
            'regime': 'Goldilocks' / 'Reflation' / 'Stagflation' / 'Deflation' / 'Unknown'
          }
        """

        growth_ft, inflation_ft = build_prometheus_feature_matrices()

        growth_now = prometheus_nowcast_single_block(
            feature_target_df=growth_ft,
            lookback_window_reg=12,
            pca_num_components_reg=10,
            lookback_window_cls=60,
            pca_num_components_cls=30
        )

        inflation_now = prometheus_nowcast_single_block(
            feature_target_df=inflation_ft,
            lookback_window_reg=12,
            pca_num_components_reg=10,
            lookback_window_cls=60,
            pca_num_components_cls=20
        )

        # Use the ensemble signs if available, else fall back to regression sign
        growth_sign = growth_now['combined_sign'] if growth_now['combined_sign'] != 0 else growth_now['reg_sign']
        inflation_sign = inflation_now['combined_sign'] if inflation_now['combined_sign'] != 0 else inflation_now[
            'reg_sign']

        regime = map_to_regime(growth_sign, inflation_sign)

        return {
            'as_of_date': max(growth_now['last_date'], inflation_now['last_date']),
            'growth': {
                'point_forecast': growth_now['reg_point_forecast'],
                'reg_sign': growth_now['reg_sign'],
                'cls_sign': growth_now['cls_sign'],
                'cls_proba_pos': growth_now['cls_proba_pos'],
                'ensemble_sign': growth_sign
            },
            'inflation': {
                'point_forecast': inflation_now['reg_point_forecast'],
                'reg_sign': inflation_now['reg_sign'],
                'cls_sign': inflation_now['cls_sign'],
                'cls_proba_pos': inflation_now['cls_proba_pos'],
                'ensemble_sign': inflation_sign
            },
            'regime': regime
        }

    def prometheus_regime_nowcast_streamlit():
        """
        Streamlit-friendly function:
          - Builds feature matrices
          - Nowcasts growth & inflation
          - Maps to regime
          - Displays results in a pretty grid
        """

        growth_ft, inflation_ft = build_prometheus_feature_matrices()

        growth_now = prometheus_nowcast_single_block(
            feature_target_df=growth_ft,
            lookback_window_reg=12,
            pca_num_components_reg=10,
            lookback_window_cls=60,
            pca_num_components_cls=30,
        )

        inflation_now = prometheus_nowcast_single_block(
            feature_target_df=inflation_ft,
            lookback_window_reg=12,
            pca_num_components_reg=10,
            lookback_window_cls=60,
            pca_num_components_cls=20,
        )

        # Choose ensemble sign when non-zero; otherwise fall back to regression sign
        growth_sign_final = (
            growth_now["ensemble_sign"] if growth_now["ensemble_sign"] != 0 else growth_now["reg_sign"]
        )
        inflation_sign_final = (
            inflation_now["ensemble_sign"] if inflation_now["ensemble_sign"] != 0 else inflation_now["reg_sign"]
        )

        regime = map_to_regime(growth_sign_final, inflation_sign_final)
        as_of_date = max(growth_now["last_date"], inflation_now["last_date"])
        as_of_str = as_of_date.strftime("%Y-%m-%d")

        # ------------------------- Streamlit layout -------------------------

        st.markdown("### Prometheus Macro Regime Nowcast")
        st.markdown(f"**As of:** `{as_of_str}`  &nbsp;&nbsp;|&nbsp;&nbsp; **Forecast Horizon:** Next Month")

        st.markdown("---")

        col1, col2, col3 = st.columns([1.2, 1.2, 1])

        # Growth column
        with col1:
            st.markdown("#### Growth Nowcast")
            st.metric(
                label="Growth 12m Δ (3m diff) – Point Forecast",
                value=f"{growth_now['reg_point_forecast']:.4f}",
            )

            growth_dir = "Improving" if growth_sign_final > 0 else "Deteriorating"
            st.write(f"**Direction:** {growth_dir}")
            st.write(
                f"- Regression sign: `{'+1' if growth_now['reg_sign'] > 0 else '-1'}`  "
                f"(point forecast {growth_now['reg_point_forecast']:.4f})"
            )
            st.write(
                f"- Classifier sign: `{'+1' if growth_now['cls_sign'] > 0 else '-1'}`  "
                f"(P(growth>0) = {growth_now['cls_proba_pos']:.2%})"
            )
            st.write(
                f"- Ensemble sign used: `{growth_sign_final:+d}`"
            )

        # Inflation column
        with col2:
            st.markdown("#### Inflation Nowcast")
            st.metric(
                label="Inflation 12m Δ (3m diff) – Point Forecast",
                value=f"{inflation_now['reg_point_forecast']:.4f}",
            )

            infl_dir = "Rising" if inflation_sign_final > 0 else "Falling"
            st.write(f"**Direction:** {infl_dir}")
            st.write(
                f"- Regression sign: `{'+1' if inflation_now['reg_sign'] > 0 else '-1'}`  "
                f"(point forecast {inflation_now['reg_point_forecast']:.4f})"
            )
            st.write(
                f"- Classifier sign: `{'+1' if inflation_now['cls_sign'] > 0 else '-1'}`  "
                f"(P(inflation>0) = {inflation_now['cls_proba_pos']:.2%})"
            )
            st.write(
                f"- Ensemble sign used: `{inflation_sign_final:+d}`"
            )

        # Regime column
        with col3:
            st.markdown("#### Predicted Regime")

            regime_color_map = {
                "Goldilocks": "#2E8B57",
                "Reflation": "#DAA520",
                "Stagflation": "#B22222",
                "Deflation": "#1E90FF",
                "Unknown": "#808080",
            }
            bg_color = regime_color_map.get(regime, "#808080")

            st.markdown(
                f"""
                <div style="
                    background-color:{bg_color};
                    padding:18px;
                    border-radius:10px;
                    text-align:center;
                    color:white;
                    font-size:20px;
                    font-weight:bold;">
                    {regime}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.write("**Grid Mapping**")
            st.write(
                "- Goldilocks: Growth ↑, Inflation ↓\n"
                "- Reflation: Growth ↑, Inflation ↑\n"
                "- Stagflation: Growth ↓, Inflation ↑\n"
                "- Deflation: Growth ↓, Inflation ↓"
            )

        # Optional: small table with core outputs (nice for download/debug)
        summary_df = pd.DataFrame(
            {
                "as_of_date": [as_of_str],
                "growth_point_forecast": [growth_now["reg_point_forecast"]],
                "growth_ensemble_sign": [growth_sign_final],
                "inflation_point_forecast": [inflation_now["reg_point_forecast"]],
                "inflation_ensemble_sign": [inflation_sign_final],
                "regime": [regime],
            }
        ).set_index("as_of_date")

        st.markdown("---")
        st.markdown("#### Nowcast Summary Table")
        st.dataframe(summary_df.style.format("{:.4f}", subset=["growth_point_forecast", "inflation_point_forecast"]))

    prometheus_regime_nowcast()


### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME OVERVIEW --------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_flowcluster_nowcast():
    print('hi')


### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME OVERVIEW --------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_crossasset_nowcast():
    print('hi')


### ---------------------------------------------------------------------------------------------------------- ###
### -------------------------------------------- REGIME OVERVIEW --------------------------------------------- ###
### ---------------------------------------------------------------------------------------------------------- ###

def plot_yc_nowcast():
    print('hi')




