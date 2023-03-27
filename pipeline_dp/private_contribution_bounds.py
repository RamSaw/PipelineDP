from dataclasses import dataclass
from functools import lru_cache
from typing import List

import pipeline_dp
from pipeline_dp import dp_computations

from pipeline_dp.histograms import Histogram


class PrivateL0Calculator:
    """Calculates differentially-private l0 bound (i.e. max_partitions_contributed)."""

    def __init__(self,
                 params: pipeline_dp.CalculatePrivateContributionBoundsParams,
                 partitions, histograms, backend) -> None:
        self._params = params
        self._backend = backend
        self._partitions = partitions
        self._histograms = histograms

    @dataclass
    class Inputs:
        l0_histogram: Histogram
        number_of_partitions: int
        possible_contribution_bounds: List[int]

    @lru_cache(maxsize=None)
    def calculate(self):
        l0_histogram = self._backend.map(
            self._histograms, lambda h: h.l0_contributions_histogram,
            "Extract l0_contributions_histogram from DatasetHistograms")
        l0_histogram = self._backend.to_multi_transformable_collection(
            l0_histogram)
        number_of_partitions = self._calculate_number_of_partitions()
        number_of_partitions = self._backend.to_multi_transformable_collection(
            number_of_partitions)
        possible_contribution_bounds = self._lower_bounds_of_bins(l0_histogram)
        possible_contribution_bounds = self._backend.to_multi_transformable_collection(
            possible_contribution_bounds)

        l0_calculation_input_col = self._backend.collect(
            [l0_histogram, number_of_partitions, possible_contribution_bounds],
            PrivateL0Calculator.Inputs,
            "Collecting L0 calculation inputs into one object")
        l0_calculation_input_col = \
            self._backend.to_multi_transformable_collection(
            l0_calculation_input_col)
        return self._backend.map(l0_calculation_input_col,
                                 lambda inputs: self._calculate_l0(inputs),
                                 "Calculate private l0 bound")

    def _calculate_l0(self, inputs: Inputs):
        scoring_function = L0ScoringFunction(self._params,
                                             inputs.number_of_partitions,
                                             inputs.l0_histogram)
        return dp_computations.ExponentialMechanism(scoring_function).apply(
            self._params.calculation_eps, inputs.possible_contribution_bounds)

    def _calculate_number_of_partitions(self):
        distinct_partitions = self._backend.distinct(
            self._partitions, "Keep only distinct partitions")
        return self._backend.size(distinct_partitions,
                                  "Calculate number of partitions")

    def _lower_bounds_of_bins(self, histogram_col):

        def histogram_to_bins(hist: Histogram):
            return list(map(lambda bin: bin.lower, hist.bins))

        return self._backend.map(histogram_col, histogram_to_bins,
                                 "Extract lowers of bins from histogram")


class L0ScoringFunction(dp_computations.ExponentialMechanism.ScoringFunction):

    def __init__(self,
                 params: pipeline_dp.CalculatePrivateContributionBoundsParams,
                 number_of_partitions: int, l0_histogram: Histogram):
        super().__init__()
        self._params = params
        self._number_of_partitions = number_of_partitions
        self._l0_histogram = l0_histogram

    def score(self, k) -> float:
        impact_noise_weight = 0.5
        return -(impact_noise_weight * self._l0_impact_noise(k) +
                 (1 - impact_noise_weight) * self._l0_impact_dropped(k))

    def _max_partitions_contributed_best_upper_bound(self):
        return min(self._params.max_partitions_contributed_upper_bound,
                   self._number_of_partitions)

    @property
    def global_sensitivity(self) -> float:
        return self._max_partitions_contributed_best_upper_bound()

    @property
    def is_monotonic(self) -> bool:
        return True

    def _l0_impact_noise(self, k):
        noise_params = dp_computations.ScalarNoiseParams(
            eps=self._params.aggregation_eps,
            delta=self._params.aggregation_delta,
            max_partitions_contributed=k,
            max_contributions_per_partition=1,
            noise_kind=self._params.aggregation_noise_kind,
            min_value=None,
            max_value=None,
            min_sum_per_partition=None,
            max_sum_per_partition=None)
        return (self._number_of_partitions *
                dp_computations.compute_dp_count_noise_std(noise_params))

    def _l0_impact_dropped(self, k):
        capped_contributions = map(
            lambda bin: max(
                min(
                    bin.lower,
                    self._max_partitions_contributed_best_upper_bound(),
                ) - k,
                0,
            ) * bin.count,
            self._l0_histogram.bins,
        )
        return sum(capped_contributions)