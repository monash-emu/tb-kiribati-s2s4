"""The Kiribati TB screening model on summer4.

Module for module, this package mirrors the original summer2 code (``code/tbh/`` in
``monash-emu/kiribati_tb_modelling``):

=========================  ============================================================
``tbh`` (summer2)          ``kiribati_tb`` (summer4)
=========================  ============================================================
``demographic_tools.py``   :mod:`kiribati_tb.demographic_tools`
``age_mixing.py``          :mod:`kiribati_tb.age_mixing`
``model.py``               :mod:`kiribati_tb.model`
``outputs.py``             :mod:`kiribati_tb.outputs`
``interventions.py``       :mod:`kiribati_tb.interventions`
``runner_tools.py``        :mod:`kiribati_tb.runner_tools`
``data/scenarios.py``      :mod:`kiribati_tb.scenarios`
=========================  ============================================================

summer4 turns on JAX's 64-bit mode when it is imported, as summer2 did, so nothing here has to.
"""
