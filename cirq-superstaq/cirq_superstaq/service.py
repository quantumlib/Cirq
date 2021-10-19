# Copyright 2021 The Cirq Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Service to access SuperstaQs API."""

import collections
import os
from typing import Any, Dict, List, Optional, Union

import applications_superstaq
import cirq
import numpy as np
import qubovert as qv
from applications_superstaq.finance import MaxSharpeOutput, MinVolOutput
from applications_superstaq.logistics import TSPOutput, WarehouseOutput
from applications_superstaq.qubo import read_json_qubo_result

import cirq_superstaq
from cirq_superstaq import job, superstaq_client


def counts_to_results(
    counter: collections.Counter, circuit: cirq.Circuit, param_resolver: cirq.ParamResolver
) -> cirq.Result:
    """Converts a collections.Counter to a cirq.Result.

    Args:
            counter: The collections.Counter of counts for the run.
            circuit: The circuit to run.
            param_resolver: A `cirq.ParamResolver` to resolve parameters in `circuit`.

        Returns:
            A `cirq.Result` for the given circuit and counter.

    """

    measurement_key_names = list(circuit.all_measurement_keys())
    measurement_key_names.sort()
    # Combines all the measurement key names into a string: {'0', '1'} -> "01"
    combine_key_names = "".join(measurement_key_names)

    samples: List[List[int]] = []
    for key in counter.keys():
        keys_as_list: List[int] = []

        # Combines the keys of the counter into a list. If key = "01", keys_as_list = [0, 1]
        for index in key:
            keys_as_list.append(int(index))

        # Gets the number of counts of the key
        # counter = collections.Counter({"01": 48, "11": 52})["01"] -> 48
        counts_of_key = counter[key]

        # Appends all the keys onto 'samples' list number-of-counts-in-the-key times
        # If collections.Counter({"01": 48, "11": 52}), [0, 1] is appended to 'samples` 48 times and
        # [1, 1] is appended to 'samples' 52 times
        for key in range(counts_of_key):
            samples.append(keys_as_list)

    result = cirq.Result(
        params=param_resolver,
        measurements={
            combine_key_names: np.array(samples),
        },
    )

    return result


class Service:
    """A class to access SuperstaQ's API.

    To access the API, this class requires a remote host url and an API key. These can be
    specified in the constructor via the parameters `remote_host` and `api_key`. Alternatively
    these can be specified by setting the environment variables `SUPERSTAQ_REMOTE_HOST` and
    `SUPERSTAQ_API_KEY`.
    """

    def __init__(
        self,
        remote_host: Optional[str] = None,
        api_key: Optional[str] = None,
        default_target: str = None,
        api_version: str = cirq_superstaq.API_VERSION,
        max_retry_seconds: int = 3600,
        verbose: bool = False,
        ibmq_token: str = None,
        ibmq_group: str = None,
        ibmq_project: str = None,
        ibmq_hub: str = None,
        ibmq_pulse: bool = True,
    ):
        """Creates the Service to access SuperstaQ's API.

        Args:
            remote_host: The location of the api in the form of an url. If this is None,
                then this instance will use the environment variable `SUPERSTAQ_REMOTE_HOST`.
                If that variable is not set, then this uses
                `flask-service.cgvd1267imk10.us-east-1.cs.amazonlightsail.com/{api_version}`,
                where `{api_version}` is the `api_version` specified below.
            api_key: A string key which allows access to the api. If this is None,
                then this instance will use the environment variable  `SUPERSTAQ_API_KEY`. If that
                variable is not set, then this will raise an `EnvironmentError`.
            default_target: Which target to default to using. If set to None, no default is set
                and target must always be specified in calls. If set, then this default is used,
                unless a target is specified for a given call. Supports either 'qpu' or
                'simulator'.
            api_version: Version of the api.
            max_retry_seconds: The number of seconds to retry calls for. Defaults to one hour.
            verbose: Whether to print to stdio and stderr on retriable errors.

        Raises:
            EnvironmentError: if the `api_key` is None and has no corresponding environment
                variable set.
        """
        self.remote_host = (
            remote_host or os.getenv("SUPERSTAQ_REMOTE_HOST") or cirq_superstaq.API_URL
        )
        self.api_key = api_key or os.getenv("SUPERSTAQ_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "Parameter api_key was not specified and the environment variable "
                "SUPERSTAQ_API_KEY was also not set."
            )
        self._client = superstaq_client._SuperstaQClient(
            remote_host=self.remote_host,
            api_key=self.api_key,
            default_target=default_target,
            api_version=api_version,
            max_retry_seconds=max_retry_seconds,
            verbose=verbose,
            ibmq_token=ibmq_token,
            ibmq_group=ibmq_group,
            ibmq_project=ibmq_project,
            ibmq_hub=ibmq_hub,
            ibmq_pulse=ibmq_pulse,
        )

    def get_counts(
        self,
        circuit: "cirq.Circuit",
        repetitions: int,
        name: Optional[str] = None,
        target: Optional[str] = None,
        param_resolver: cirq.ParamResolverOrSimilarType = cirq.ParamResolver({}),
    ) -> collections.Counter:
        """Runs the given circuit on the SuperstaQ API and returns the result
        of the ran circuit as a collections.Counter

        Args:
            circuit: The circuit to run.
            repetitions: The number of times to run the circuit.
            name: An optional name for the created job. Different from the `job_id`.
            target: Where to run the job. Can be 'qpu' or 'simulator'.
            param_resolver: A `cirq.ParamResolver` to resolve parameters in  `circuit`.

        Returns:
            A `collection.Counter` for running the circuit.
        """
        resolved_circuit = cirq.protocols.resolve_parameters(circuit, param_resolver)
        counts = self.create_job(resolved_circuit, repetitions, name, target).counts()

        return counts

    def run(
        self,
        circuit: "cirq.Circuit",
        repetitions: int,
        name: Optional[str] = None,
        target: Optional[str] = None,
        param_resolver: cirq.ParamResolver = cirq.ParamResolver({}),
    ) -> cirq.Result:
        """Run the given circuit on the SuperstaQ API and returns the result
        of the ran circut as a cirq.Result.

        Args:
            circuit: The circuit to run.
            repetitions: The number of times to run the circuit.
            name: An optional name for the created job. Different from the `job_id`.
            target: Where to run the job. Can be 'qpu' or 'simulator'.
            param_resolver: A `cirq.ParamResolver` to resolve parameters in  `circuit`.

        Returns:
            A `cirq.Result` for running the circuit.
        """
        counts = self.get_counts(circuit, repetitions, name, target, param_resolver)
        return counts_to_results(counts, circuit, param_resolver)

    def create_job(
        self,
        circuit: cirq.Circuit,
        repetitions: int = 1000,
        name: Optional[str] = None,
        target: Optional[str] = None,
    ) -> job.Job:
        """Create a new job to run the given circuit.

        Args:
            circuit: The circuit to run.
            repetitions: The number of times to repeat the circuit. Defaults to 100.
            name: An optional name for the created job. Different from the `job_id`.
            target: Where to run the job. Can be 'qpu' or 'simulator'.

        Returns:
            A `cirq_superstaq.Job` which can be queried for status or results.

        Raises:
            SuperstaQException: If there was an error accessing the API.
        """
        serialized_program = cirq_superstaq.serialization.serialize_circuits(circuit)
        result = self._client.create_job(
            serialized_program=serialized_program, repetitions=repetitions, target=target, name=name
        )
        # The returned job does not have fully populated fields, so make
        # a second call and return the results of the fully filled out job.
        return self.get_job(result["job_id"])

    def get_job(self, job_id: str) -> job.Job:
        """Gets a job that has been created on the SuperstaQ API.

        Args:
            job_id: The UUID of the job. Jobs are assigned these numbers by the server during the
            creation of the job.

        Returns:
            A `cirq_superstaq.Job` which can be queried for status or results.

        Raises:
            SuperstaQNotFoundException: If there was no job with the given `job_id`.
            SuperstaQException: If there was an error accessing the API.
        """
        job_dict = self._client.get_job(job_id=job_id)
        return job.Job(client=self._client, job_dict=job_dict)

    def get_balance(self, pretty_output: bool = True) -> Union[str, float]:
        """Get the querying user's account balance in USD.

        Args:
            pretty_output: whether to return a pretty string or a float of the balance.

        Returns:
            If pretty_output is True, returns the balance as a nicely formatted string ($-prefix,
                commas on LHS every three digits, and two digits after period). Otherwise, simply
                returns a float of the balance.
        """
        balance = self._client.get_balance()["balance"]
        if pretty_output:
            return f"${balance:,.2f}"
        return balance

    def aqt_compile(
        self, circuits: Union[cirq.Circuit, List[cirq.Circuit]]
    ) -> "cirq_superstaq.aqt.AQTCompilerOutput":
        """Compiles the given circuit(s) to AQT device, optimized to its native gate set.

        Args:
            circuits: cirq Circuit(s) with operations on qubits 4 through 8.
        Returns:
            object whose .circuit(s) attribute is an optimized cirq Circuit(s)
            If qtrl is installed, the object's .seq attribute is a qtrl Sequence object of the
            pulse sequence corresponding to the optimized cirq.Circuit(s) and the
            .pulse_list(s) attribute is the list(s) of cycles.
        """
        if isinstance(circuits, cirq.Circuit):
            serialized_program = cirq_superstaq.serialization.serialize_circuits([circuits])
            circuits_list = False
        else:
            serialized_program = cirq_superstaq.serialization.serialize_circuits(circuits)
            circuits_list = True

        json_dict = self._client.aqt_compile(serialized_program)

        from cirq_superstaq import aqt

        return aqt.read_json(json_dict, circuits_list)

    def ibmq_compile(self, circuit: cirq.Circuit, target: Optional[str] = None) -> Any:
        """Returns pulse schedule for the given circuit and target.

        Qiskit must be installed for returned object to correctly deserialize to a pulse schedule.
        """
        serialized_program = cirq_superstaq.serialization.serialize_circuits([circuit])
        json_dict = self._client.ibmq_compile(serialized_program, target)
        try:
            return applications_superstaq.converters.deserialize(json_dict["pulses"])[0]
        except ModuleNotFoundError as e:
            raise cirq_superstaq.SuperstaQModuleNotFoundException(
                name=str(e.name), context="ibmq_compile"
            )

    def submit_qubo(self, qubo: qv.QUBO, target: str, repetitions: int = 1000) -> np.recarray:
        """Submits the given QUBO to the target backend. The result of the optimization
        is returned to the user as a numpy.recarray.

        Args:
            qubo: Qubovert QUBO object representing the optimization problem.
            target: A string indicating which backend to use.
            repetitions: Number of shots to execute on the device.
        Returns:
            Numpy.recarray containing the solution to the QUBO, the energy of the
            different solutions, and the number of times each solution was found.
        """
        json_dict = self._client.submit_qubo(qubo, target, repetitions=repetitions)
        return read_json_qubo_result(json_dict)

    def find_min_vol_portfolio(
        self,
        stock_symbols: List[str],
        desired_return: float,
        years_window: float = 5.0,
        solver: str = "anneal",
    ) -> MinVolOutput:
        """Finds the portfolio with minimum volatility that exceeds a specified desired return.

        Args:
            stock_symbols: A list of stock tickers to pick from.
            desired_return: The minimum return needed.
            years_window: The number of years previous from today to pull data from
            for price data.
            solver: Specifies which solver to use. Defaults to a simulated annealer.

        Returns:
            MinVolOutput object, with the following attributes:
            .best_portfolio: The assets in the optimal portfolio.
            .best_ret: The return of the optimal portfolio.
            .best_std_dev: The volatility of the optimal portfolio.

        """
        input_dict = {
            "stock_symbols": stock_symbols,
            "desired_return": desired_return,
            "years_window": years_window,
            "solver": solver,
        }
        json_dict = self._client.find_min_vol_portfolio(input_dict)
        from applications_superstaq import finance

        return finance.read_json_minvol(json_dict)

    def find_max_pseudo_sharpe_ratio(
        self,
        stock_symbols: List[str],
        k: float,
        num_assets_in_portfolio: int = None,
        years_window: float = 5.0,
        solver: str = "anneal",
    ) -> MaxSharpeOutput:
        """
        Finds the optimal equal-weight portfolio from a possible pool of stocks
        according to the following rules:
        -All stock must come from the stock_symbols list.
        -All stocks will be equally weighted in the portfolio.
        -The "pseudo" Sharpe ratio of the portfolio is maximized.

        The Sharpe ratio can be thought of as the ratio of reward to risk.
        The formula for the Sharpe ratio is the portfolio's expected return less the risk-free
        rate divided by the portfolio standard deviation. For the risk-free rate, we will use the
        three month treasury bill rate. Instead of maximizing the Sharpe ratio directly, we will
        minimize variance minus return net the risk-free rate. The user specifies a factor k, as
        describes below to favor reducing risk or favor increasing expected return, each likely
        at the expense of the other. The Sharpe ratio of the resulting portfolio is returned,
        since it is relevant information.

        To summarize, we optimize:
        k * standard_deviation_expression - (1 - k) * expected_return_expression


        Args:
            stock_symbols: A list of stock tickers to pick from.
            k: A risk factor coefficient between 0 and 1. A k closer to 1
            indicates only being concerned with risk aversion, while a k closer to 0
            indicates only being concerned with maximizing expected return regardless of
            risk.
            k: The factor to weigh the portions of the expression.
            num_assets_in_portfolio: The number of desired assets in the portfolio.
            If not specified, then the function will iterate through and
            check for all portfolio sizes.
            years_window: The number of years previous from today to pull data from
            for price data.
            solver: Specifies which solver to use. Defaults to a simulated annealer.

        Return:
            A MaxSharpeOutput object with the following attributes:
            .best_portfolio: The assets in the optimal portfolio.
            .best_ret: The return of the optimal portfolio.
            .best_std_dev: The volatility of the optimal portfolio.
            .best_sharpe_ratio: The Sharpe ratio of the optimal portfolio.

        """
        input_dict = {
            "stock_symbols": stock_symbols,
            "k": k,
            "num_assets_in_portfolio": num_assets_in_portfolio,
            "years_window": years_window,
            "solver": solver,
        }
        json_dict = self._client.find_max_pseudo_sharpe_ratio(input_dict)
        from applications_superstaq import finance

        return finance.read_json_maxsharpe(json_dict)

    def tsp(self, locs: List[str], solver: str = "anneal") -> TSPOutput:
        """
        This function solves the traveling salesperson problem (TSP) and
        takes a list of strings as input. TSP finds the shortest tour that
        traverses all locations in a list.
        Each string should be an addresss or name of a landmark
        that can pinpoint a location as a Google Maps search.
        It is assumed that the first string in the list is
        the starting and ending point for the TSP tour.
        The function returns a dictionary containing the route,
        the indices of the route from the input list, and the total distance
        of the tour in miles.

        Args:
            locs: List of strings where each string represents
            a location needed to be visited on tour.
            solver: A string indicating which solver to use ("rqaoa" or "anneal").

        Returns:
            A TSPOutput object with the following attributes:
            .route: The optimal TSP tour as a list of strings in order.
            .route_list_numbers: The indicies in locs of the optimal tour.
            .total_distance: The tour's total distance.
            .map_links: A link to google maps that show the tour.

        """
        input_dict = {"locs": locs}
        json_dict = self._client.tsp(input_dict)
        from applications_superstaq import logistics

        return logistics.read_json_tsp(json_dict)

    def warehouse(
        self, k: int, possible_warehouses: List[str], customers: List[str], solver: str = "anneal"
    ) -> WarehouseOutput:
        """
        This function solves the warehouse location problem, which is:
        given a list of customers to be served and  a list of possible warehouse
        locations, find the optimal k warehouse locations such that the sum of
        the distances to each customer from the nearest facility is minimized.

        Args:
            k: An integer representing the number of warehouses in the solution.
            possible_warehouses: A list of possible warehouse locations.
            customers: A list of customer locations.
            solver: A string indicating which solver to use ("rqaoa" or "anneal").

        Returns:
            A WarehouseOutput object with the following attributes:
            .warehouse_to_destination: The optimal warehouse-customer pairings in List(Tuple) form.
            .total_distance: The tour's total distance among all warehouse-customer pairings.
            .map_link: A link to google maps that show the tour.
            .open_warehouses: A list of all warehouses that are open.

        """
        input_dict = {
            "k": k,
            "possible_warehouses": possible_warehouses,
            "customers": customers,
            "solver": solver,
        }
        json_dict = self._client.warehouse(input_dict)
        from applications_superstaq import logistics

        return logistics.read_json_warehouse(json_dict)

    def aqt_upload_configs(self, pulses_file_path: str, variables_file_path: str) -> Dict[str, str]:
        """Uploads configs for AQT

        Args:
            pulses_file_path: The filepath for Pulses.yaml
            variables_file_path: The filepath for Variables.yaml
        Returns:
            A dictionary of of the status of the update (Whether or not it failed)
        """
        with open(pulses_file_path) as pulses_file:
            read_pulses = pulses_file.read()

        with open(variables_file_path) as variables_file:
            read_variables = variables_file.read()

        json_dict = self._client.aqt_upload_configs(
            {"pulses": read_pulses, "variables": read_variables}
        )

        return json_dict
