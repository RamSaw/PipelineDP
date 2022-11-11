# Copyright 2022 OpenMined.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Demo of running PipelineDP locally, without any external data processing framework"""
from typing import NamedTuple

from absl import app
from absl import flags
import pipeline_dp
from collections import defaultdict
import statistics

from common_utils import parse_file, write_to_file

FLAGS = flags.FLAGS
flags.DEFINE_string('input_file', None, 'The file with the movie view data')
flags.DEFINE_string('output_file', None, 'Output file')


class MetricsTuple(NamedTuple):
    count: int
    sum: int
    privacy_id_count: int


def main(unused_argv):
    # Here, we use a local backend for computations. This does not depend on
    # any pipeline framework and it is implemented in pure Python in
    # PipelineDP. It keeps all data in memory and is not optimized for large data.
    # For datasets smaller than ~tens of megabytes, local execution without any
    # framework is faster than local mode with Beam or Spark.
    backend = pipeline_dp.LocalBackend()

    # Define the privacy budget available for our computation.
    budget_accountant = pipeline_dp.NaiveBudgetAccountant(total_epsilon=1,
                                                          total_delta=1e-6)

    # Load and parse input data
    movie_views = parse_file(FLAGS.input_file)

    calculate_exact_values(movie_views)

    # Create a DPEngine instance.
    dp_engine = pipeline_dp.DPEngine(budget_accountant, backend)

    contribution_bounds_params = pipeline_dp.ContributionBoundsDpCalculationParameters(
        max_partitions_contributed_upper_bound=100,
        max_contributions_per_partition=1,
        budget_weight=0.1
    )

    params = pipeline_dp.AggregateParams(
        metrics=[
            # we can compute multiple metrics at once.
            pipeline_dp.Metrics.COUNT,
            pipeline_dp.Metrics.SUM,
            pipeline_dp.Metrics.PRIVACY_ID_COUNT
        ],
        budget_weight=0.9,
        # .. with minimal rating of "1"
        min_value=1,
        # .. and maximum rating of "5"
        max_value=5,
        public_partitions=list(range(1, 9)))

    # Specify how to extract privacy_id, partition_key and value from an
    # element of movie_views.
    data_extractors = pipeline_dp.DataExtractors(
        partition_extractor=lambda mv: mv.movie_id,
        privacy_id_extractor=lambda mv: mv.user_id,
        value_extractor=lambda mv: mv.rating)

    params.max_partitions_contributed, params.max_contributions_per_partition = dp_engine.calculate_contribution_bounds(movie_views, contribution_bounds_params, params, data_extractors)

    # Create a computational graph for the aggregation.
    # All computations are lazy. dp_result is iterable, but iterating it would
    # fail until budget is computed (below).
    # It’s possible to call DPEngine.aggregate multiple times with different
    # metrics to compute.
    dp_result = dp_engine.aggregate(movie_views, params, data_extractors)

    budget_accountant.compute_budgets()

    # Here's where the lazy iterator initiates computations and gets transformed
    # into actual results
    dp_result = list(dp_result)

    # Save the results
    with open(FLAGS.output_file, 'w') as out:
        out.write("max_partitions_contributed = " + str(params.max_partitions_contributed))
        out.write("\nmax_contributions_per_partition = " + str(params.max_contributions_per_partition) + "\n\n")
    dp_result = sorted(dp_result, key=lambda el: el[0])
    write_to_file(dp_result, FLAGS.output_file)

    with open(FLAGS.output_file, "r") as out:
        print(out.read())
    return 0


def calculate_exact_values(movie_views):
    movies_partitioned = defaultdict(list)
    partitions_contributed = defaultdict(set)
    contributions_per_partition = defaultdict(int)
    for movie_view in movie_views:
        movies_partitioned[movie_view.movie_id].append(movie_view)
        partitions_contributed[movie_view.user_id].add(movie_view.movie_id)
        contributions_per_partition[(movie_view.user_id, movie_view.movie_id)] += 1
    results = []
    for movie, movie_view_list in movies_partitioned.items():
        results.append((movie, MetricsTuple(
            len(movie_view_list),
            sum(map(lambda movie_view: movie_view.rating, movie_view_list)),
            len(set(map(lambda movie_view: movie_view.user_id, movie_view_list))))))
    with open("/Users/mpravilov/Documents/datasets/netflix_sampled/exact_output.txt", "w") as f:
        f.write("median_partitions_contributed = " + str(
            statistics.median(map(lambda user_movies: len(user_movies), partitions_contributed.values()))) + "\n")
        f.write("average_partitions_contributed = " + str(
            statistics.mean(map(lambda user_movies: len(user_movies), partitions_contributed.values()))) + "\n")
        f.write("max_partitions_contributed = " + str(
            max(map(lambda user_movies: len(user_movies), partitions_contributed.values()))))
        f.write("\nmax_contributions_per_partition = " + str(max(contributions_per_partition.values())) + "\n\n")
        f.write("\n".join(map(lambda el: str(el), results)))


if __name__ == '__main__':
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    app.run(main)
