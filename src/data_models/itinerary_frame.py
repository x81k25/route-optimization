"""
ItineraryFrame: Simple DataFrame subclass for route optimization itinerary data.

Extends pl.DataFrame to enforce the correct schema for itinerary objects.
"""

from typing import Dict, Any
import polars as pl
from datetime import datetime


class ItineraryFrame(pl.DataFrame):
    """
    DataFrame subclass for itinerary data with enforced schema.

    Schema matches README specification for itinerary.jsonl:
    - zone_id, day, pos_id, pos_name, pos_class, route, action,
      schedule, duration, route_order, clusterer, router, balancer, created_on
    """

    EXPECTED_SCHEMA = {
        "zone_id": pl.Utf8,
        "day": pl.Int64,
        "pos_id": pl.Utf8,
        "pos_name": pl.Utf8,
        "pos_class": pl.Utf8,
        "route": pl.List(pl.List(pl.Float64)),
        "action": pl.Utf8,
        "schedule": pl.Float64,
        "duration": pl.Float64,
        "route_order": pl.Int64,
        "clusterer": pl.Utf8,
        "router": pl.Utf8,
        "balancer": pl.Utf8,
        "created_on": pl.Utf8
    }

    @classmethod
    def from_assignments(
        cls,
        zone_id: str,
        assignments: Dict[str, Any],
        zone_df: pl.DataFrame,
        model_params: Dict[str, Any],
        clusterer: str = None,
        router: str = None,
        balancer: str = None
    ):
        """
        Create ItineraryFrame from assignment data.

        :param zone_id: Zone identifier
        :param assignments: Dictionary with primary_assignments and secondary_assignments
        :param zone_df: Zone location data
        :param model_params: Model parameters configuration dictionary
        :param clusterer: Clustering algorithm
        :param router: Routing algorithm
        :param balancer: Balancing method
        :return: ItineraryFrame instance
        """
        records = []
        created_on = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Calculate durations from model parameters
        primary_hours_per_week = model_params["primary_hours_per_week"]
        hours_per_non_primary = model_params["hours_per_non_primary"]
        secondary_duration_minutes = hours_per_non_primary * 60  # convert to minutes

        # For primary duration, we need to calculate per day per primary
        # We'll calculate this individually for each primary based on their day assignments

        # Add primary assignments
        primary_day_durations = assignments.get("primary_day_durations", {})
        for pos_id, days in assignments.get("primary_assignments", {}).items():
            pos_data = zone_df.filter(pl.col("pos_id") == pos_id).row(0, named=True)

            for day in days:
                # Get the specific duration for this primary on this day
                duration_minutes = primary_day_durations.get((pos_id, day), 0.0)

                records.append({
                    "zone_id": zone_id,
                    "day": day,
                    "pos_id": str(pos_id),
                    "pos_name": pos_data["name"],
                    "pos_class": "primary",
                    "route": [[pos_data["longitude"], pos_data["latitude"]]],  # Include coordinates
                    "action": None,  # Will be populated in later stages
                    "schedule": None,  # Will be populated in later stages
                    "duration": duration_minutes,
                    "route_order": None,  # Will be populated in route optimization stage
                    "clusterer": clusterer,
                    "router": router,
                    "balancer": balancer,
                    "created_on": created_on
                })

        # Add secondary assignments
        for pos_id, days in assignments.get("secondary_assignments", {}).items():
            pos_data = zone_df.filter(pl.col("pos_id") == pos_id).row(0, named=True)
            for day in days:
                records.append({
                    "zone_id": zone_id,
                    "day": day,
                    "pos_id": str(pos_id),
                    "pos_name": pos_data["name"],
                    "pos_class": "secondary",
                    "route": [[pos_data["longitude"], pos_data["latitude"]]],  # Include coordinates
                    "action": None,  # Will be populated in later stages
                    "schedule": None,  # Will be populated in later stages
                    "duration": secondary_duration_minutes,
                    "route_order": None,  # Will be populated in route optimization stage
                    "clusterer": clusterer,
                    "router": router,
                    "balancer": balancer,
                    "created_on": created_on
                })

        if records:
            df = pl.DataFrame(records, schema=cls.EXPECTED_SCHEMA)
        else:
            df = pl.DataFrame(schema=cls.EXPECTED_SCHEMA)

        return cls._from_pydf(df._df)