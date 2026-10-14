"""Combine-to-NFL translation modeling pipeline."""

from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from bdb_2027.config import Config


class TranslationPipeline:
    """Predict regular-season NFL outcomes from Combine sensor traits."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()

    def train_cv(
        self,
        features_df: pl.DataFrame,
        feature_cols: list[str],
        target_col: str,
        group_col: str = "nfl_id",
        model_type: str = "lgbm",
    ) -> dict[str, Any]:
        """Train and evaluate models using GroupKFold to prevent player data leakage."""
        df_clean = features_df.filter(
            pl.col(target_col).is_not_null()
            & pl.all_horizontal([pl.col(c).is_not_null() for c in feature_cols])
        )

        min_samples = self.config.n_splits * 2
        if df_clean.height < min_samples:
            raise ValueError(
                f"Insufficient samples ({df_clean.height}) for "
                f"{self.config.n_splits}-fold CV."
            )

        X = df_clean.select(feature_cols).to_numpy()
        y = df_clean.select(target_col).to_series().to_numpy()
        groups = df_clean.select(group_col).to_series().to_numpy()

        gkf = GroupKFold(n_splits=self.config.n_splits)
        oof_preds = np.zeros(len(y))
        feature_importances = np.zeros(len(feature_cols))
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            if model_type == "rf":
                model = RandomForestRegressor(
                    n_estimators=self.config.lgb_n_estimators or 50,
                    max_depth=self.config.lgb_max_depth or 3,
                    random_state=self.config.seed + fold,
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                fold_imp = model.feature_importances_
            elif model_type == "gb":
                model = GradientBoostingRegressor(
                    n_estimators=self.config.lgb_n_estimators or 40,
                    max_depth=2,
                    learning_rate=self.config.lgb_learning_rate or 0.05,
                    random_state=self.config.seed + fold,
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                fold_imp = model.feature_importances_
            elif model_type == "ridge":
                model = Ridge(alpha=5.0)
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                fold_imp = np.abs(model.coef_)
                if np.sum(fold_imp) > 0:
                    fold_imp = fold_imp / np.sum(fold_imp)
            elif model_type == "ensemble":
                m_rf = RandomForestRegressor(
                    n_estimators=50, max_depth=3, random_state=self.config.seed + fold
                )
                m_ridge = Ridge(alpha=5.0)
                m_rf.fit(X_train, y_train)
                m_ridge.fit(X_train, y_train)
                preds = 0.5 * m_rf.predict(X_val) + 0.5 * m_ridge.predict(X_val)
                fold_imp = m_rf.feature_importances_
            else:
                model = lgb.LGBMRegressor(
                    n_estimators=self.config.lgb_n_estimators,
                    learning_rate=self.config.lgb_learning_rate,
                    max_depth=self.config.lgb_max_depth,
                    random_state=self.config.seed + fold,
                    verbosity=-1,
                )
                model.fit(
                    X_train,
                    y_train,
                    eval_X=X_val,
                    eval_y=y_val,
                    callbacks=[
                        lgb.early_stopping(
                            stopping_rounds=self.config.early_stopping_patience,
                            verbose=False,
                        )
                    ],
                )
                preds = model.predict(X_val)
                fold_imp = model.feature_importances_

            oof_preds[val_idx] = preds
            feature_importances += fold_imp / self.config.n_splits

            fold_r2 = r2_score(y_val, preds)
            fold_mse = mean_squared_error(y_val, preds)
            fold_metrics.append({"fold": fold, "r2": fold_r2, "mse": fold_mse})

        overall_r2 = r2_score(y, oof_preds)
        overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
        pearson_corr, _ = pearsonr(y, oof_preds)
        spearman_corr, _ = spearmanr(y, oof_preds)

        importance_dict = {
            col: float(imp)
            for col, imp in zip(feature_cols, feature_importances, strict=False)
        }
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
        )

        return {
            "target": target_col,
            "overall_r2": float(overall_r2),
            "overall_rmse": float(overall_rmse),
            "pearson_r": float(pearson_corr),
            "spearman_r": float(spearman_corr),
            "fold_metrics": fold_metrics,
            "feature_importance": sorted_importance,
            "oof_predictions": oof_preds,
            "actuals": y,
        }
