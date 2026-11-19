# API reference

```{eval-rst}
.. automodule:: kiribati_tb.model
   :members: get_tb_model, TBModel, get_natural_tb_model, get_force_of_infection, tanh_based_scaleup, add_detection_and_treatment, stratify_model_by_age, stratify_model_by_reachability, add_births_and_deaths, NegTxOutcomes, get_neg_tx_outcomes, get_neg_tx_outcome_funcs

.. automodule:: kiribati_tb.model_summer4
   :members: get_tb_model, screening_flows, flow_sources, death_outputs, chain, reverse

.. automodule:: kiribati_tb.model_strata
   :members: get_tb_model, compartment, get_force_of_infection, screening_flows, flow_sources, death_outputs

.. automodule:: kiribati_tb.outputs
   :members: request_model_outputs, request_per_capita_output

.. automodule:: kiribati_tb.variants
   :members: Variant

.. automodule:: kiribati_tb.runner_tools
   :members: get_parameters_and_priors, midpoint_params, run_model, run_compartments, compartment_labels

.. automodule:: kiribati_tb.age_mixing
   :members: get_model_ready_age_mixing_matrix, build_mixing_stack, yearly_matrices, read_conmat_matrix, canberra_distance

.. automodule:: kiribati_tb.interventions
   :members: ScreeningTools, ScreeningProgram, Scenario, scr_rate_func

.. automodule:: kiribati_tb.calibration
   :members: get_bcm, get_targets, get_priors, normal_target

.. automodule:: kiribati_tb.comparison
   :members: solver_comparison, annual_flow_comparison, cumulative_comparison
```
