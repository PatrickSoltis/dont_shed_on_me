# dont_shed_on_me

## Abstract

Electricity is essential to daily life in the developed world, powering critical systems and services such as hospitals, water supply and wastewater treatment, and other functions. Power outages--such as those driven by increasingly large wildfires and Public Safety Power Shutoffs in California--compromise functionality of these critical services. Microgrids that include storage and distributed generation resources can help alleviate some of these stresses, with the ability to isolate or ‘island’ from the main power grid, and distribute power locally. However, microgrids typically have limited storage and generation available, therefore the ability to prioritize loads and optimize discharge schedules can help to maximize the benefit that these resources can provide, and minimize harm. This study aims to create a model that produces an optimal storage dispatch schedule based on the relative priority of serving different loads, and storage and distributed generation resources available, in order to maximize the benefit of energy storage.

## Repository Contents

* MicrogridModel0_1.py - Original model
* MicrogridModel0_2.py - Latest version
* git_testing.txt - Place for the team to practice git commands

## Significant updates in Latest Version (0.2)

* All generation is an apparent power (S) term. (This lets s_max and relaxed apparent power definition cap generation at realistic value.)
* As a result, battery charging (real term b_eat) and discharging (apparent term b_gen) are now in separate terms. Major weakness is that b_eat and b_gen terms currently can both be active in same time step.
* Model is optimized for fraction of real power delivered (F_P), since apparent power delivered no longer exists.
* Constraints for P, Q, and L set terms corresponding to line 0-0 to 0. Before, they erroneously set all terms at time step 0 to 0.
