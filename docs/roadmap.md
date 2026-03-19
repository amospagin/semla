# Roadmap

## Shipped

- [x] ML estimation with lavaan-validated results
- [x] MLR estimator (robust Satorra-Bentler / Yuan-Bentler-Mplus)
- [x] DWLS estimator for ordinal data (polychoric correlations)
- [x] FIML for missing data
- [x] Bayesian MCMC estimation via NumPyro (NUTS sampler)
- [x] GPU-accelerated Bayesian inference via JAX
- [x] Batch Bayesian estimation with CPU+GPU parallel scheduling
- [x] Data-adaptive and weak informative priors
- [x] Adaptive convergence monitoring
- [x] WAIC and PSIS-LOO model comparison
- [x] IRT models (1PL, 2PL, GRM) with ICC, information functions, ability estimation
- [x] Multi-group CFA (configural, metric, scalar, strict invariance)
- [x] Growth curve models (linear and nonlinear)
- [x] Higher-order (second-order) factor models
- [x] Cross-lagged panel models
- [x] Standardized solutions (std.all, std.lv)
- [x] Modification indices (Schur complement score test)
- [x] Chi-square difference test
- [x] Multi-model comparison table (compare_models)
- [x] AIC / BIC / adjusted BIC
- [x] R-squared for endogenous variables
- [x] Mean structure and intercepts (~1)
- [x] Equality constraints via parameter labels
- [x] Nonlinear parameter constraints (>, <, >=, <=, ==)
- [x] Indirect effects and mediation (:= operator with delta method SEs)
- [x] Reliability measures (McDonald's omega, Cronbach's alpha)
- [x] Factor score prediction (regression and Bartlett)
- [x] Bootstrap confidence intervals
- [x] Residual diagnostics and Mardia's multivariate normality test
- [x] Model-implied matrices (fitted)
- [x] Parameter variance-covariance matrix (vcov)
- [x] Parallel chain execution on CPU
- [x] Positive loading constraints for sign identification
- [x] Automated measurement invariance testing (measurementInvariance)
- [x] Auto-add exogenous latent variable covariances in sem() (lavaan-matching)
- [x] True DWLS objective function (diagonally weighted least squares)
- [x] FIML validated on multiple MCAR missingness patterns
- [x] Validated against lavaan 0.6-20

## Future Directions

- [ ] vmap-batched MCMC for same-structure models ([#55](https://github.com/amospagin/semla/issues/55))
- [ ] Bayesian IRT estimation ([#48](https://github.com/amospagin/semla/issues/48))
- [ ] Multilevel SEM for clustered data ([#49](https://github.com/amospagin/semla/issues/49))
- [ ] Latent class analysis and mixture models ([#50](https://github.com/amospagin/semla/issues/50))
- [ ] ESEM — exploratory SEM ([#51](https://github.com/amospagin/semla/issues/51))
- [ ] Monte Carlo simulation / power analysis ([#52](https://github.com/amospagin/semla/issues/52))
- [ ] Validate RI-CLPM ([#53](https://github.com/amospagin/semla/issues/53))
- [ ] Path diagram visualization ([#38](https://github.com/amospagin/semla/issues/38))
- [ ] Publication-ready table export ([#46](https://github.com/amospagin/semla/issues/46))
- [ ] Complex survey weight support ([#47](https://github.com/amospagin/semla/issues/47))
- [ ] Alignment method for approximate invariance ([#54](https://github.com/amospagin/semla/issues/54))
- [ ] PyPI release
