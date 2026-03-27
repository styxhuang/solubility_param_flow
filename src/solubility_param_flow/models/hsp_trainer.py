"""Traditional ML trainer for HSP prediction benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from solubility_param_flow.models.feature_builder import HSPFeatureBuilder


class TraditionalMLTrainer:
    """Train and compare traditional ML models on HSP targets."""

    def __init__(self, random_state: int = 2026):
        self.random_state = random_state
        self.feature_builder = HSPFeatureBuilder()

    def run_benchmark(
        self,
        dataset_path: str,
        output_dir: str,
        feature_sets: list[str] | None = None,
    ) -> pd.DataFrame:
        feature_sets = feature_sets or ["rdkit", "cosmo", "combined"]
        base_dir = Path(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        frame = self.feature_builder.load_success_rows(dataset_path)
        metrics_rows: list[dict[str, object]] = []
        prediction_frames: list[pd.DataFrame] = []

        for feature_set in feature_sets:
            x_frame, y_frame = self.feature_builder.build_feature_frame(frame, feature_set)
            result = self._train_feature_set(feature_set, x_frame, y_frame)
            metrics_rows.extend(result["metrics"])
            prediction_frames.append(result["predictions"])

        metrics_frame = pd.DataFrame(metrics_rows)
        predictions_frame = pd.concat(prediction_frames, ignore_index=True)

        metrics_frame.to_csv(base_dir / "metrics_summary.csv", index=False)
        predictions_frame.to_csv(base_dir / "predictions.csv", index=False)
        self._write_feature_manifest(base_dir / "feature_manifest.json", feature_sets)
        self._plot_metrics(metrics_frame, base_dir / "model_comparison.png")
        self._plot_best_predictions(predictions_frame, metrics_frame, base_dir / "best_model_scatter.png")

        return metrics_frame

    def _train_feature_set(
        self,
        feature_set: str,
        x_frame: pd.DataFrame,
        y_frame: pd.DataFrame,
    ) -> dict[str, object]:
        from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        if len(x_frame) < 4:
            raise ValueError(
                f"Feature set `{feature_set}` needs at least 4 successful rows for train/test split."
            )

        test_size = 0.25 if len(x_frame) >= 8 else 0.5
        x_train, x_test, y_train, y_test = train_test_split(
            x_frame,
            y_frame,
            test_size=test_size,
            random_state=self.random_state,
        )

        models = {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(
                n_estimators=200,
                random_state=self.random_state,
            ),
            "extra_trees": ExtraTreesRegressor(
                n_estimators=300,
                random_state=self.random_state,
            ),
        }

        metrics_rows: list[dict[str, object]] = []
        prediction_rows: list[pd.DataFrame] = []
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            prediction = model.predict(x_test)

            mae = mean_absolute_error(y_test, prediction, multioutput="uniform_average")
            rmse = mean_squared_error(y_test, prediction, multioutput="uniform_average") ** 0.5
            r2 = r2_score(y_test, prediction, multioutput="uniform_average")

            metrics_rows.append(
                {
                    "feature_set": feature_set,
                    "model": model_name,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "n_train": len(x_train),
                    "n_test": len(x_test),
                }
            )

            prediction_frame = pd.DataFrame(prediction, columns=y_test.columns, index=y_test.index)
            prediction_frame = prediction_frame.add_prefix("pred_")
            joined = pd.concat([y_test.reset_index(drop=True), prediction_frame.reset_index(drop=True)], axis=1)
            joined["feature_set"] = feature_set
            joined["model"] = model_name
            prediction_rows.append(joined)

        return {
            "metrics": metrics_rows,
            "predictions": pd.concat(prediction_rows, ignore_index=True),
        }

    def _write_feature_manifest(self, target_path: Path, feature_sets: list[str]) -> None:
        payload = {
            "feature_sets": feature_sets,
            "rdkit_columns": self.feature_builder.rdkit_columns,
            "cosmo_columns": self.feature_builder.cosmo_columns,
            "target_columns": self.feature_builder.target_columns,
        }
        target_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def _plot_metrics(metrics_frame: pd.DataFrame, output_path: Path) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        metric_frame = metrics_frame.melt(
            id_vars=["feature_set", "model"],
            value_vars=["mae", "rmse", "r2"],
            var_name="metric",
            value_name="value",
        )

        plt.figure(figsize=(12, 6))
        sns.barplot(data=metric_frame, x="feature_set", y="value", hue="model", ci=None)
        plt.title("Traditional ML Benchmark by Feature Set")
        plt.ylabel("Metric Value")
        plt.xlabel("Feature Set")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def _plot_best_predictions(
        predictions_frame: pd.DataFrame,
        metrics_frame: pd.DataFrame,
        output_path: Path,
    ) -> None:
        import matplotlib.pyplot as plt

        best_row = metrics_frame.sort_values("mae", ascending=True).iloc[0]
        best_predictions = predictions_frame[
            (predictions_frame["feature_set"] == best_row["feature_set"])
            & (predictions_frame["model"] == best_row["model"])
        ]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for axis, target in zip(axes, ["delta_d", "delta_p", "delta_h"]):
            pred_column = f"pred_{target}"
            axis.scatter(best_predictions[target], best_predictions[pred_column], alpha=0.8)
            min_value = min(best_predictions[target].min(), best_predictions[pred_column].min())
            max_value = max(best_predictions[target].max(), best_predictions[pred_column].max())
            axis.plot([min_value, max_value], [min_value, max_value], linestyle="--")
            axis.set_title(target)
            axis.set_xlabel("True")
            axis.set_ylabel("Predicted")

        fig.suptitle(
            f"Best Model: {best_row['model']} | Feature Set: {best_row['feature_set']}"
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
