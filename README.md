# PILCO: A Model-Based Policy Search Framework (Python Port)

This is a Python port of the PILCO (Probabilistic Inference for Learning Control) framework, originally implemented in MATLAB. PILCO is a data-efficient, model-based reinforcement learning method for continuous state and action spaces.

## Quick Start

Run the cart-pole scenario:

```
cd pilco_python/scenarios/cartPole
python cartPole_learn.py
```

## Project Structure

- `/scenarios/`- Example environments (cart-pole, etc.)

- `/core/`- Core PILCO algorithms

- `/models/`- Probabilistic dynamics models

- `/controllers/`- Policy implementations

- `/util/`- Utilities and helpers

## Dependencies

- Python 3.7+

- NumPy

- SciPy

- Matplotlib

Install with:

```
pip install -r requirements.txt
```

## Features

- Exact Bayesian inference for policy evaluation

- Probabilistic dynamics modeling with Gaussian Processes

- Gradient-based policy optimization

- Data-efficient learning (typically < 5 episodes)

## Create Your Own Scenario

1. Copy an existing scenario from `/scenarios/`

2. Implement your dynamics model

3. Define your reward function

4. Configure training parameters

## Reference

If you use this code, please cite the original PILCO paper:

```
@inproceedings{deisenroth2011pilco,
  title={PILCO: A model-based and data-efficient approach to policy search},
  author={Deisenroth, Marc Peter and Rasmussen, Carl Edward},
  booktitle={International Conference on Machine Learning},
  year={2011}
}
```

## Notes

- This is a community port of the original MATLAB implementation

- Results may differ slightly from the MATLAB version

## Original Authors

Marc Deisenroth, Andrew McHutchon, Joe Hall, Carl Edward Rasmussen

## License

Same as pilco-matlab license, check LICENSE.txt for more.
