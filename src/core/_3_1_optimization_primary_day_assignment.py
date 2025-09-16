"""
Stage 3.1: Primary Day Assignment
Assign primary (high-value) locations to dedicated days
"""

# standard library imports
from typing import Dict, List, Tuple

# 3rd-party imports
from loguru import logger
import polars as pl
import yaml

# local imports
from src.data_models import ItineraryFrame


def assign_primary_days(
    zone_df: pl.DataFrame,
    model_params: dict
) -> ItineraryFrame:
    """
    Assign primary locations to dedicated days using simple division logic.

    This is substage 3.1 where we:
    1. Get number of primary stores for the zone
    2. Divide primary_hours_per_week by hours_per_day to get primary days needed
    3. Distribute primary stores across those days (no drive time consideration)
    4. Calculate remaining secondary days for clustering

    :param zone_df: Zone-specific location DataFrame
    :param model_params: Model parameters configuration dictionary
    :return: Itinerary DataFrame with day assignments
    """
    zone_id = zone_df["zone_id"].unique().to_list()[0]  # Extract zone_id from data
    logger.info(f"stage 3.1: primary day assignment - zone {zone_id}")

    # use passed model parameters
    days_per_week = model_params["days_per_week"]
    hours_per_day = model_params["hours_per_day"]
    primary_hours_per_week = model_params["primary_hours_per_week"]

    # separate primary and secondary locations
    primary_df = zone_df.filter(pl.col("class") == "primary")
    secondary_df = zone_df.filter(pl.col("class") == "secondary")

    num_primary = len(primary_df)
    logger.info(f"found {num_primary} primary, {len(secondary_df)} secondary locations")

    # calculate primary days needed using simple division (no drive time)
    if num_primary > 0:
        hours_per_primary = primary_hours_per_week / num_primary
        days_per_primary = hours_per_primary / hours_per_day
        total_primary_days = num_primary * days_per_primary
        logger.info(f"each primary gets {hours_per_primary:.1f} hours = {days_per_primary:.2f} days")
        logger.info(f"total primary days needed: {total_primary_days:.2f}")
    else:
        total_primary_days = 0
        logger.info("no primary stores - all days become secondary")

    # calculate secondary days using the correct formula from instructions
    # secondary_days = (days_per_week) - (sum of primary hours in store)/hours_per_day
    # round to nearest whole number
    primary_hours_total = primary_hours_per_week if num_primary > 0 else 0
    primary_days_used = primary_hours_total / hours_per_day if hours_per_day > 0 else 0
    secondary_days = round(days_per_week - primary_days_used)

    # Ensure non-negative
    if secondary_days < 0:
        secondary_days = 0

    logger.info(f"primary hours total: {primary_hours_total}, primary days used: {primary_days_used:.2f}")
    logger.info(f"secondary days available: {secondary_days}")

    # assign primary stores to days
    primary_assignments = {}  # pos_id -> [list of days]
    primary_day_durations = {}  # (pos_id, day) -> duration_in_minutes

    if num_primary > 0:
        # For distributing primary stores across days:
        # Example: 2 stores × 24 hours ÷ 8 hours/day = 3 total days
        # Store 1 gets days 1,2 and Store 2 gets days 2,3 (day 2 is shared)

        if num_primary == 1:
            # Single primary gets all its days consecutively
            days_needed = int(round(primary_hours_per_week / hours_per_day))
            pos_id = primary_df.row(0, named=True)["pos_id"]
            primary_assignments[pos_id] = list(range(1, days_needed + 1))

            # Set full day durations for single primary
            full_day_minutes = hours_per_day * 60
            for day in range(1, days_needed + 1):
                primary_day_durations[(pos_id, day)] = full_day_minutes

            logger.info(f"single primary {pos_id} assigned to days: {list(range(1, days_needed + 1))}")

        elif num_primary == 2:
            # Two primaries: need to handle fractional days correctly
            hours_per_primary = primary_hours_per_week / 2
            days_per_primary = hours_per_primary / hours_per_day
            primary_ids = [row["pos_id"] for row in primary_df.iter_rows(named=True)]

            if days_per_primary == 1.5:
                # Special case: 1.5 days each = 3 days total
                # Primary 1: full day 1 + half day 2 = 1.5 days
                # Primary 2: half day 2 + full day 3 = 1.5 days
                # Day 2 is shared between both primaries (each gets 4 hours)
                primary_assignments[primary_ids[0]] = [1, 2]  # Primary 1: day 1 (8hrs) + day 2 (4hrs) = 12hrs total
                primary_assignments[primary_ids[1]] = [2, 3]  # Primary 2: day 2 (4hrs) + day 3 (8hrs) = 12hrs total

                # Set durations: full days get hours_per_day, shared days get half
                full_day_minutes = hours_per_day * 60
                half_day_minutes = (hours_per_day / 2) * 60
                primary_day_durations[(primary_ids[0], 1)] = full_day_minutes  # Primary 1, day 1: full day
                primary_day_durations[(primary_ids[0], 2)] = half_day_minutes  # Primary 1, day 2: half day
                primary_day_durations[(primary_ids[1], 2)] = half_day_minutes  # Primary 2, day 2: half day
                primary_day_durations[(primary_ids[1], 3)] = full_day_minutes  # Primary 2, day 3: full day

                logger.info(f"primary {primary_ids[0]} assigned 1.5 days: full day 1 + half day 2")
                logger.info(f"primary {primary_ids[1]} assigned 1.5 days: half day 2 + full day 3")
                logger.info(f"day 2 is shared: {half_day_minutes/60} hours each primary")
            elif days_per_primary < 1.0:
                # Both primaries share day 1
                primary_assignments[primary_ids[0]] = [1]
                primary_assignments[primary_ids[1]] = [1]

                # Set shared durations: each gets their portion of the day
                shared_minutes = days_per_primary * hours_per_day * 60
                primary_day_durations[(primary_ids[0], 1)] = shared_minutes
                primary_day_durations[(primary_ids[1], 1)] = shared_minutes

                logger.info(f"primary {primary_ids[0]} assigned to days: [1]")
                logger.info(f"primary {primary_ids[1]} assigned to days: [1]")
            else:
                # General case: round to nearest integer
                days_needed = int(round(days_per_primary))
                primary_assignments[primary_ids[0]] = list(range(1, days_needed + 1))
                primary_assignments[primary_ids[1]] = list(range(days_needed, 2 * days_needed))

                # Set full day durations for general case
                full_day_minutes = hours_per_day * 60
                for day in range(1, days_needed + 1):
                    primary_day_durations[(primary_ids[0], day)] = full_day_minutes
                for day in range(days_needed, 2 * days_needed):
                    primary_day_durations[(primary_ids[1], day)] = full_day_minutes

                logger.info(f"primary {primary_ids[0]} assigned to days: {list(range(1, days_needed + 1))}")
                logger.info(f"primary {primary_ids[1]} assigned to days: {list(range(days_needed, 2 * days_needed))}")

        elif num_primary == 3:
            # Three primaries: each gets 8 hours = 1 day each
            day = 1
            full_day_minutes = hours_per_day * 60
            for row in primary_df.iter_rows(named=True):
                pos_id = row["pos_id"]
                primary_assignments[pos_id] = [day]
                primary_day_durations[(pos_id, day)] = full_day_minutes
                logger.info(f"primary {pos_id} assigned to day: {day}")
                day += 1

        else:
            # General case: distribute evenly
            hours_per_primary = primary_hours_per_week / num_primary
            days_per_primary = hours_per_primary / hours_per_day

            current_day = 1
            full_day_minutes = hours_per_day * 60
            for row in primary_df.iter_rows(named=True):
                pos_id = row["pos_id"]
                days_needed = max(1, int(round(days_per_primary)))
                assigned_days = []
                for _ in range(days_needed):
                    if current_day <= days_per_week:
                        assigned_days.append(current_day)
                        primary_day_durations[(pos_id, current_day)] = full_day_minutes
                        current_day += 1
                primary_assignments[pos_id] = assigned_days
                logger.info(f"primary {pos_id} assigned to days: {assigned_days}")

    # assign secondary stores to remaining days (dumb split)
    secondary_assignments = {}  # pos_id -> [list of days]

    if len(secondary_df) > 0 and secondary_days > 0:
        # calculate which days are available for secondary assignments
        primary_used_days = set()
        for days_list in primary_assignments.values():
            primary_used_days.update(days_list)

        available_days = [day for day in range(1, days_per_week + 1) if day not in primary_used_days]

        # if we calculated more secondary days than available, use available days
        if secondary_days > len(available_days):
            secondary_days = len(available_days)
            logger.warning(f"adjusted secondary days to {secondary_days} based on available days: {available_days}")
        else:
            # take only the first secondary_days from available days
            available_days = available_days[:secondary_days]

        logger.info(f"available days for secondaries: {available_days}")

        # dumb split: distribute secondary stores evenly across available days
        secondary_stores = [row["pos_id"] for row in secondary_df.iter_rows(named=True)]
        stores_per_day = len(secondary_stores) // len(available_days) if available_days else 0
        extra_stores = len(secondary_stores) % len(available_days) if available_days else 0

        store_index = 0
        for day_index, day in enumerate(available_days):
            stores_for_this_day = stores_per_day + (1 if day_index < extra_stores else 0)
            day_stores = []

            for _ in range(stores_for_this_day):
                if store_index < len(secondary_stores):
                    store_id = secondary_stores[store_index]
                    secondary_assignments[store_id] = [day]
                    day_stores.append(store_id)
                    store_index += 1

            if day_stores:
                logger.info(f"day {day}: assigned {len(day_stores)} secondary stores: {day_stores}")

    logger.success(f"primary assignment complete: {secondary_days} days available for secondary clustering")
    logger.success(f"secondary assignment complete: {len(secondary_assignments)} stores assigned to {secondary_days} days")

    # Create ItineraryFrame using the new class
    assignments = {
        "primary_assignments": primary_assignments,
        "secondary_assignments": secondary_assignments,
        "primary_day_durations": primary_day_durations
    }

    itinerary = ItineraryFrame.from_assignments(
        zone_id=zone_id,
        assignments=assignments,
        zone_df=zone_df,
        model_params=model_params
    )

    logger.info(f"created ItineraryFrame with {len(itinerary)} records")
    return itinerary