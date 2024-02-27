import psycopg2
from datetime import datetime

from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    TargetDriftPreset,
    RegressionPreset,
    DataQualityPreset,
)
from evidently import ColumnMapping
from evidently.metrics import *
import logging

logging.basicConfig(level=logging.INFO)

import os

from dotenv import load_dotenv

load_dotenv("/Users/alexander.girardet/Code/Personal/projects/rightmove_project/.env")

PG_URI = os.environ.get("MONITORING_URI_PG")


class MetricExtraction:
    def __init__(self):
        self.conn = None
        self.cur = None

    def connect_to_postgres(self):
        self.conn = psycopg2.connect(dsn=PG_URI)
        self.cur = self.conn.cursor()

    def close_connection(self):
        self.conn.close()

    def extract_data_quality(self, quality_report):
        # Initialize a dictionary to store the results
        summary_dict = {
            "walk_score": {},
            "price": {},
            "bedrooms": {},
            "bathrooms": {},
        }

        nans_by_columns = {}

        for metric in quality_report["metrics"]:
            if metric["metric"] == "DatasetSummaryMetric":
                nans_by_columns = metric["result"]["current"]["nans_by_columns"]
                continue

            if "column_name" in metric["result"]:
                column_name = metric["result"]["column_name"]

                if column_name in summary_dict:
                    summary_dict[column_name]["reference_mean"] = metric["result"][
                        "reference_characteristics"
                    ]["mean"]
                    summary_dict[column_name]["current_mean"] = metric["result"][
                        "current_characteristics"
                    ]["mean"]
                    summary_dict[column_name]["current_count"] = metric["result"][
                        "current_characteristics"
                    ]["count"]
                    summary_dict[column_name]["current_nulls"] = nans_by_columns.get(
                        column_name, 0
                    )

        return summary_dict

    def extract_drift(self, drift_report):
        share_of_drifted_columns = drift_report["metrics"][0]["result"][
            "share_of_drifted_columns"
        ]
        dataset_drift_binary = drift_report["metrics"][0]["result"]["dataset_drift"]
        target_drift_score = drift_report["metrics"][1]["result"]["drift_by_columns"][
            "target"
        ]["drift_score"]
        target_drift_detected = drift_report["metrics"][1]["result"][
            "drift_by_columns"
        ]["target"]["drift_detected"]

        summary_dict = {
            "share_of_drifted_columns": share_of_drifted_columns,
            "dataset_drift_binary": dataset_drift_binary,
            "target_drift_score": target_drift_score,
            "target_drift_detected": target_drift_detected,
        }

        return summary_dict

    def extract_performance(self, performance_report):
        reference_r2 = performance_report["metrics"][0]["result"]["reference"][
            "r2_score"
        ]
        reference_rmse = performance_report["metrics"][0]["result"]["reference"]["rmse"]
        reference_mean_error = performance_report["metrics"][0]["result"]["reference"][
            "mean_error"
        ]
        reference_mean_abs_error = performance_report["metrics"][0]["result"][
            "reference"
        ]["mean_abs_error"]

        current_r2 = performance_report["metrics"][0]["result"]["current"]["r2_score"]
        current_rmse = performance_report["metrics"][0]["result"]["current"]["rmse"]
        current_mean_error = performance_report["metrics"][0]["result"]["current"][
            "mean_error"
        ]
        current_mean_abs_error = performance_report["metrics"][0]["result"]["current"][
            "mean_abs_error"
        ]

        summary_dict = {
            "reference": {
                "r2": reference_r2,
                "rmse": reference_rmse,
                "mean_error": reference_mean_error,
                "mean_abs_error": reference_mean_abs_error,
            },
            "current": {
                "r2": current_r2,
                "rmse": current_rmse,
                "mean_error": current_mean_error,
                "mean_abs_error": current_mean_abs_error,
            },
        }

        return summary_dict

    def extract_prediction(self, prediction_report):
        prediction_drift_score = prediction_report["metrics"][0]["result"][
            "drift_score"
        ]
        prediction_drift_detected = prediction_report["metrics"][0]["result"][
            "drift_detected"
        ]

        summary_dict = {
            "prediction_drift_score": prediction_drift_score,
            "prediction_drift_detected": prediction_drift_detected,
        }

        return summary_dict

    def load_metrics_to_postgres(
        self, data_dict, metric_category, loading_timestamp=None
    ):
        insert_query = """
        INSERT INTO model_metrics (metric_category, metric_name, metric_value, metric_status, created_at)
        VALUES (%s, %s, %s, %s, %s)
        """

        if self.cur is None:
            self.connect_to_postgres()

        if loading_timestamp:
            current_timestamp = loading_timestamp
        else:
            current_timestamp = datetime.now()

        for key, value in data_dict.items():
            if isinstance(
                value, dict
            ):  # For nested dictionaries like in 'extract_means'
                for sub_key, sub_value in value.items():
                    # Determine if sub_value is a boolean and assign appropriately
                    if isinstance(sub_value, bool):
                        self.cur.execute(
                            insert_query,
                            (
                                metric_category,
                                f"{key}_{sub_key}",
                                None,
                                sub_value,
                                current_timestamp,
                            ),
                        )
                    else:
                        self.cur.execute(
                            insert_query,
                            (
                                metric_category,
                                f"{key}_{sub_key}",
                                sub_value,
                                None,
                                current_timestamp,
                            ),
                        )
            else:
                # Check if the value is boolean and assign to metric_status instead of metric_value
                if isinstance(value, bool):
                    self.cur.execute(
                        insert_query,
                        (metric_category, key, None, value, current_timestamp),
                    )
                else:
                    # Assuming all non-dict and non-boolean values should be treated as numeric
                    self.cur.execute(
                        insert_query,
                        (metric_category, key, value, None, current_timestamp),
                    )

        self.conn.commit()

    def get_target_drift_metrics(self, current_data, reference_data):
        target_drift_report = Report(metrics=[ColumnDriftMetric("target")])

        target_drift_report.run(
            reference_data=reference_data, current_data=current_data
        )

        predict_drift_report_dict = target_drift_report.as_dict()

        prediction_data = self.extract_prediction(predict_drift_report_dict)
        return prediction_data

    def get_performance_metrics(self, current_data, reference_data):
        reg_performance_report = Report(
            metrics=[
                RegressionQualityMetric(),
            ]
        )

        reg_performance_report.run(
            reference_data=reference_data, current_data=current_data
        )

        reg_performance_dict = reg_performance_report.as_dict()

        performance_data = self.extract_performance(reg_performance_dict)
        return performance_data

    def get_data_drift_metrics(self, current_data, reference_data):
        data_drift_report = Report(
            metrics=[
                DataDriftPreset(),
            ]
        )

        data_drift_report.run(reference_data=reference_data, current_data=current_data)

        data_drift_report_dict = data_drift_report.as_dict()

        drift_data = self.extract_drift(data_drift_report_dict)
        return drift_data

    def get_data_quality_metrics(self, current_data, reference_data):
        column_mapping = ColumnMapping()

        current_data = current_data[["bedrooms", "bathrooms", "walk_score", "target"]]
        reference_data = reference_data[
            ["bedrooms", "bathrooms", "walk_score", "target"]
        ]

        numerical_features = ["bedrooms", "bathrooms", "walk_score"]

        column_mapping.numerical_features = numerical_features
        column_mapping.target = "target"

        data_quality_report = Report(metrics=[DataQualityPreset()])

        data_quality_report.run(
            current_data=current_data, reference_data=reference_data
        )

        data_quality_report_dict = data_quality_report.as_dict()

        quality_data = self.extract_data_quality(data_quality_report_dict)
        return quality_data
