"""Position rules engine — maps observations + MC output to position signals."""
from collections import defaultdict
from datetime import datetime, UTC

from hormuz.db import HormuzDB
from hormuz.models import MCResult, Observation, PositionSignal


class PositionEngine:
    """Evaluate position rules against observations and MC results.

    Two rule categories:
    1. HARD STOP-LOSS — always checked, cannot be overridden
    2. MC-DRIVEN — conditional on attack frequency, transit volume, etc.
    """

    def __init__(self, db: HormuzDB, position_config: dict):
        self.db = db
        self.config = position_config
        self.stop_loss = position_config.get("stop_loss", {})

    def evaluate(
        self,
        observations: list[Observation] | None = None,
        mc_result: MCResult | None = None,
    ) -> list[PositionSignal]:
        """Evaluate all rules, return triggered signals with executed=False."""
        if not observations:
            return []

        now = datetime.now(UTC)
        signals: list[PositionSignal] = []

        # --- HARD STOP-LOSS ---
        signals.extend(self._check_brent_stop(observations, now))
        signals.extend(self._check_portfolio_loss(observations, now))

        # --- MC-DRIVEN RULES ---
        signals.extend(self._check_high_attack_low_transit(observations, now))
        signals.extend(self._check_low_attack(observations, now))
        signals.extend(self._check_transit_partial_recovery(observations, now))
        signals.extend(self._check_transit_full_recovery(observations, now))

        return signals

    # -- helpers --

    @staticmethod
    def _daily_values(observations: list[Observation], key: str) -> dict[str, float]:
        """Group observations by date, take the latest value per day for a given key.

        Returns {date_str: value} sorted by date.
        """
        by_date: dict[str, list[Observation]] = defaultdict(list)
        for obs in observations:
            if obs.key == key:
                date_str = obs.timestamp.strftime("%Y-%m-%d")
                by_date[date_str].append(obs)

        result = {}
        for date_str in sorted(by_date):
            # take the last observation of the day (by timestamp)
            latest = max(by_date[date_str], key=lambda o: o.timestamp)
            result[date_str] = latest.value
        return result

    @staticmethod
    def _max_consecutive_meeting(daily: dict[str, float], predicate) -> int:
        """Count the max consecutive days (from the tail) where predicate holds."""
        values = list(daily.values())  # already sorted by date
        if not values:
            return 0
        # Check from end backwards for consecutive streak
        count = 0
        for v in reversed(values):
            if predicate(v):
                count += 1
            else:
                break
        return count

    # -- HARD STOP-LOSS rules --

    def _check_brent_stop(self, obs: list[Observation], now: datetime) -> list[PositionSignal]:
        brent_below = self.stop_loss.get("brent_below", 80)
        required_days = self.stop_loss.get("days", 3)

        daily = self._daily_values(obs, "brent_price")
        consecutive = self._max_consecutive_meeting(daily, lambda v: v < brent_below)

        if consecutive >= required_days:
            return [PositionSignal(
                timestamp=now,
                trigger=f"brent_stop_loss: <${brent_below} for {consecutive} consecutive days",
                action="平掉全部能源超配",
                executed=False,
            )]
        return []

    def _check_portfolio_loss(self, obs: list[Observation], now: datetime) -> list[PositionSignal]:
        max_loss = self.config.get("max_loss_pct", 8)

        daily = self._daily_values(obs, "portfolio_loss_pct")
        if not daily:
            return []

        # Check the latest value
        latest_loss = list(daily.values())[-1]
        if latest_loss > max_loss:
            return [PositionSignal(
                timestamp=now,
                trigger=f"portfolio_loss_stop: {latest_loss}% > {max_loss}%",
                action="全部头寸减半",
                executed=False,
            )]
        return []

    # -- MC-DRIVEN rules --

    def _check_high_attack_low_transit(
        self, obs: list[Observation], now: datetime
    ) -> list[PositionSignal]:
        """Attack freq > 3/day for 7 consecutive days AND transit < 3 mbd -> add energy to 22%."""
        attack_daily = self._daily_values(obs, "attack_frequency")
        transit_daily = self._daily_values(obs, "transit_volume")

        attack_consecutive = self._max_consecutive_meeting(attack_daily, lambda v: v > 3.0)
        transit_consecutive = self._max_consecutive_meeting(transit_daily, lambda v: v < 3.0)

        if attack_consecutive >= 7 and transit_consecutive >= 7:
            return [PositionSignal(
                timestamp=now,
                trigger="high_attack_low_transit: attack>3/day x7d AND transit<3mbd x7d",
                action="能源加仓至22%",
                executed=False,
            )]
        return []

    def _check_low_attack(self, obs: list[Observation], now: datetime) -> list[PositionSignal]:
        """Attack freq < 1/day for 5 consecutive days -> adjust irgcDecayMean."""
        daily = self._daily_values(obs, "attack_frequency")
        consecutive = self._max_consecutive_meeting(daily, lambda v: v < 1.0)

        if consecutive >= 5:
            return [PositionSignal(
                timestamp=now,
                trigger="low_attack_freq: <1/day for 5 consecutive days",
                action="irgcDecayMean下调至4周（不减仓）",
                executed=False,
            )]
        return []

    def _check_transit_partial_recovery(
        self, obs: list[Observation], now: datetime
    ) -> list[PositionSignal]:
        """Transit > 8 mbd for 3 days -> unwind half energy overweight."""
        daily = self._daily_values(obs, "transit_volume")
        consecutive = self._max_consecutive_meeting(daily, lambda v: v > 8.0)

        if consecutive >= 3:
            return [PositionSignal(
                timestamp=now,
                trigger="transit_partial_recovery: >8mbd for 3 consecutive days",
                action="平掉能源超配1/2",
                executed=False,
            )]
        return []

    def _check_transit_full_recovery(
        self, obs: list[Observation], now: datetime
    ) -> list[PositionSignal]:
        """Transit > 12 mbd for 5 days -> unwind all overweight, back to base."""
        daily = self._daily_values(obs, "transit_volume")
        consecutive = self._max_consecutive_meeting(daily, lambda v: v > 12.0)

        if consecutive >= 5:
            return [PositionSignal(
                timestamp=now,
                trigger="transit_full_recovery: >12mbd for 5 consecutive days",
                action="平掉全部超配，回基准",
                executed=False,
            )]
        return []
