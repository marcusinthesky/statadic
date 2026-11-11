"""Reproduce feat-docs' actually tested spatial estimator pairs in CPU float64."""
import numpy as np

ROOT = 'src/python/packages/statadic/tests/'
CROSS = ROOT + 'estimation/test_spatial_reference.py'
DIAG = ROOT + 'inference/test_spatial_diagnostics_reference.py'
GMM = ROOT + 'estimation/spatial/test_gmm_reference.py'
PANEL = ROOT + 'estimation/spatial/test_panel.py'
KKP = ROOT + 'estimation/spatial/test_kkp_reference.py'
OBS = ROOT + 'estimation/spatial/test_observed_panel.py'


def row(method, reference, fixture, work, source, rtol, atol, note=''):
    return dict(method=method, reference=reference, fixture=fixture, work=work,
                source=source, note=note, rtol=rtol, atol=atol)


CASES = {}
for model, name, impl in [('sar', 'SAR', 'ML_Lag'), ('sem', 'SEM', 'ML_Error')]:
    for durbin in (False, True):
        key = model + ('_durbin' if durbin else '')
        CASES[key] = row(name + (' + Durbin covariates' if durbin else ' ML'), 'spreg.' + impl,
                         '40 nodes, 2 covariates, seed 126', 'Coefficients, log likelihood and covariance',
                         CROSS, 1e-5, 2e-6, 'spreg public estimator defaults, including its ancillary outputs; prepared weights reused on both sides.')
for robust in (False, True):
    CASES['spatial_iv' + ('_white' if robust else '')] = row('Spatial lag IV' + (' (White)' if robust else ' (classical)'),
        'spreg.GM_Lag', '40 nodes, 2 covariates, seed 126', 'Coefficients and covariance', CROSS, 1e-7, 1e-8)
CASES['slx'] = row('SLX least squares (HC1)', 'statsmodels.OLS', '40 nodes, 2 covariates, seed 126',
    'Coefficients and HC1 covariance', CROSS, 1e-7, 1e-10, 'Prepares graph on statadic side; constructs lagged design in timed call on both sides.')
CASES['moran_lm'] = row('OLS residual Moran and LM tests', 'spreg.OLS', '30 nodes, 2 covariates, seed 126',
    'Fit OLS, residual Moran and 5 LM tests', DIAG, 1e-8, 1e-10, 'spreg computes Moran in its fitted OLS object; avoids duplicate MoranRes invocation.')
CASES['ak'] = row('Spatial IV + Anselin–Kelejian test', 'spreg.GM_Lag + AKtest', '40 nodes, 2 covariates, seed 126',
    'Fit and AK statistic/p-value', DIAG, 1e-8, 1e-10)
CASES['cfh'] = row('Spatial Durbin + common-factor test', 'spreg.ML_Lag', '40 nodes, 2 covariates, seed 126',
    'Fit, covariance and common-factor statistic/p-value', DIAG, 1e-5, 1e-7)
for lag in (False, True):
    for het in (False, True):
        key = ('gmm_combo' if lag else 'gmm_error') + ('_het' if het else '_hom')
        impl = ('GM_Combo' if lag else 'GM_Error') + ('_Het' if het else '_Hom')
        CASES[key] = row('GS2SLS ' + ('lag + error' if lag else 'error') + (' (heteroskedastic)' if het else ' (homoskedastic)'),
            'spreg.' + impl, '60 nodes, 2 covariates, seed 126', 'Coefficients and covariance, one update', GMM, 3e-4, 2e-5)
for model, name, impl in [('sar', 'SAR', 'Panel_FE_Lag'), ('sem', 'SEM', 'Panel_FE_Error')]:
    CASES['panel_fe_' + model] = row('Unit FE spatial panel ' + name, 'spreg.' + impl,
        '12 units × 8 periods, 2 covariates, seed 53', 'Coefficients, normalized log likelihood and covariance',
        PANEL, 2e-5, 2e-6)
    CASES['observed_fe_' + model] = row('Observed-panel unit FE ' + name + ' + WX', 'spreg.' + impl,
        '8 units × 5 periods, 1 covariate, seed 534', 'Coefficients and normalized log likelihood', OBS, 1e-7, 3e-5,
        'Only the completely observed case is comparable. spreg receives explicit WX, assembled during fit.')
for full in (False, True):
    CASES['kkp' + ('_full' if full else '')] = row('KKP spatial random effects' + (' (full weights)' if full else ''),
        'spreg.GM_KKP', '30 units × 8 periods, 2 covariates, seed 126',
        'Coefficients, spatial lambda, variance components and conditional covariance', KKP, 2e-4, 2e-5,
        'Reference optimizer stopping criteria tightened exactly as in parity test.')


def pack(*values):
    return np.concatenate([np.asarray(v).ravel() for v in values])


def cross_fixture(case):
    rng = np.random.default_rng(126)
    gmm = case.startswith('gmm_')
    n = 60 if gmm else (30 if case == 'moran_lm' else 40)
    weight = 0.6 if gmm or case in ('ak', 'cfh') else 0.8
    w = weight * np.roll(np.eye(n), 1, axis=1) + (1-weight) * np.roll(np.eye(n), -1, axis=1)
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    if gmm:
        shock = rng.normal(size=n) * (0.3 + 0.2*np.abs(x[:,1]) if case.endswith('_het') else 0.5)
        y = x @ np.array([0.3, -0.5, 0.8]) + np.linalg.solve(np.eye(n)-0.3*w, shock)
        if case.startswith('gmm_combo'):
            y = np.linalg.solve(np.eye(n)-0.25*w, y)
    elif case == 'moran_lm':
        y = rng.normal(size=n)
    elif case in ('ak','cfh'):
        y = np.linalg.solve(np.eye(n)-0.3*w, x @ np.array([0.3,-0.5,0.8]) + rng.normal(size=n)*0.5)
    else:
        y = np.linalg.solve(np.eye(n)-0.35*w, x @ np.array([0.4,-0.5,0.8]) + rng.normal(size=n)*0.4)
    return x,y,w


def prepare(case, backend):
    if case.startswith(('panel_fe_', 'observed_fe_', 'kkp')):
        return prepare_panel(case, backend)
    x,y,w = cross_fixture(case)
    gmm = case.startswith('gmm_')
    lag = case.startswith('gmm_combo')
    het = case.endswith('_het')
    durbin = case.endswith('durbin') or case == 'cfh'
    if backend == 'statadic':
        from statadic.estimation.spatial import ML, SAR, SEM, SpatialIV, SpatialProblem, SpatialGMM, SLXLeastSquares
        from statadic.inference.spatial import spatial_covariance
        from statadic.operators.spatial import prepare_spatial_weights
        problem = SpatialProblem(x, y, prepare_spatial_weights(w))
        if case == 'moran_lm':
            from statadic.estimation.linear import LeastSquares, LinearProblem
            from statadic.inference.spatial.diagnostics import spatial_diagnostics
            def run():
                diag = spatial_diagnostics(LeastSquares().estimate(LinearProblem(x,y)), problem.weights)
                return pack(*[(getattr(diag,k).statistic,getattr(diag,k).pvalue) for k in
                    ('lm_lag','lm_error','robust_lm_lag','robust_lm_error','lm_sarma','moran')])
            return run
        if case == 'slx':
            from statadic.inference.linear_covariance import hc_covariance
            def run():
                fit = SLXLeastSquares((1,2)).estimate(problem)
                return pack(fit.parameters,hc_covariance(fit,kind='HC1').covariance.values)
            return run
        if gmm:
            estimator = SpatialGMM(variance_model='heteroskedastic' if het else 'homoskedastic',
                spatial_lag_model=lag,instrument_columns=(1,2),tolerance=1e-9)
        elif case.startswith('spatial_iv') or case=='ak':
            estimator = SpatialIV(instrument_columns=(1,2))
        else:
            estimator = ML((SEM if case.startswith('sem') else SAR)(slx_columns=(1,2) if durbin else ()))
        def run():
            fit = estimator.estimate(problem)
            if case == 'ak':
                from statadic.inference.spatial import spatial_ak_test
                stat = spatial_ak_test(fit)
                return pack(stat.statistic,stat.pvalue)
            if case.startswith('spatial_iv'):
                cov = spatial_covariance(fit,method='heteroskedastic' if case.endswith('white') else 'homoskedastic')
            else:
                cov = spatial_covariance(fit)
            if case == 'cfh':
                from statadic.inference.spatial import spatial_common_factor_test
                stat = spatial_common_factor_test(cov)
                return pack(stat.statistic,stat.pvalue)
            if gmm or case.startswith('spatial_iv'):
                return pack(fit.parameters,cov.covariance.values)
            return pack(fit.parameters,fit.loglik,cov.covariance.values)
        return run
    if case == 'slx':
        import statsmodels.api as sm
        def run():
            fit = sm.OLS(y,np.column_stack([x,w@x[:,1:]])).fit(cov_type='HC1',use_t=True)
            return pack(fit.params,fit.cov_params())
        return run
    import spreg
    from libpysal.weights import full2W
    graph = full2W(w)
    def run():
        if case == 'moran_lm':
            fit = spreg.OLS(y[:,None],x[:,1:],w=graph,spat_diag=True,moran=True)
            return pack(fit.lm_lag,fit.lm_error,fit.rlm_lag,fit.rlm_error,fit.lm_sarma,(fit.moran_res[0],fit.moran_res[2]))
        if gmm:
            cls = getattr(spreg,CASES[case]['reference'].split('.')[1])
            kwargs = dict(w_lags=1) if lag else {}
            fit = cls(y[:,None],x[:,1:],w=graph,epsilon=1e-9,max_iter=1,**kwargs)
            return pack(fit.betas,fit.vm)
        if case.startswith('spatial_iv') or case=='ak':
            fit = spreg.GM_Lag(y[:,None],x[:,1:],w=graph,w_lags=1,robust='white' if case.endswith('white') else None,sig2n_k=False)
            if case=='ak':
                stat = spreg.AKtest(fit,graph,case='gen')
                return pack(stat.ak,stat.p)
            return pack(fit.betas,fit.vm)
        cls = spreg.ML_Error if case.startswith('sem') else spreg.ML_Lag
        fit = cls(y[:,None],x[:,1:],graph,slx_lags=int(durbin),epsilon=1e-9)
        if case=='cfh':
            return pack(fit.cfh_test)
        return pack(fit.betas,fit.logll,fit.vm)
    return run
def prepare_panel(case, backend):
    kkp = case.startswith('kkp')
    observed = case.startswith('observed_fe_')
    sar = case.endswith('_sar')
    t,n,p = (8,30,3) if kkp else (5,8,1) if observed else (8,12,2)
    rng = np.random.default_rng(126 if kkp else 534 if observed else 53)
    weight = 0.6 if kkp else 0.5
    w = weight*np.roll(np.eye(n),1,axis=1) + (1-weight)*np.roll(np.eye(n),-1,axis=1)
    if kkp:
        x = np.concatenate([np.ones((t,n,1)),rng.normal(size=(t,n,2))],axis=2)
        errors = np.linalg.solve(np.eye(n)-0.3*w,(rng.normal(size=n)+rng.normal(size=(t,n))*0.5).T).T
        y = x @ np.array([0.3,-0.5,0.8]) + errors
    else:
        x = rng.normal(size=(t,n,p))
        alpha = rng.normal(size=n)
        noise = rng.normal(size=(t,n))*0.4
        mean = (x[:,:,0]*0.5 + np.einsum('ij,tjp->tip',w,x)[:,:,0]*0.2 if observed else x @ np.array([0.5,-0.4])) + alpha
        y = np.linalg.solve(np.eye(n)-(0.2 if observed else 0.3)*w,(mean+noise).T).T if sar else mean + np.linalg.solve(np.eye(n)+0.2*w,noise.T).T
    if backend == 'statadic':
        from statadic.estimation.spatial import KKP, PanelML, SAR, SEM, SpatialPanelProblem, ObservedSpatialPanelProblem, ObservedPanelML
        from statadic.operators.spatial import prepare_spatial_panel_weights, prepare_spatial_weights, ObservedSpatialPeriod
        if observed:
            graph = prepare_spatial_weights(w)
            periods = tuple(ObservedSpatialPeriod(graph,tuple(range(n)),tuple(range(n))) for _ in range(t))
            problem = ObservedSpatialPanelProblem(periods,tuple(x),tuple(y),n)
            estimator = ObservedPanelML((SAR if sar else SEM)((0,)),effects='unit')
            def run():
                fit = estimator.estimate(problem)
                return pack(fit.parameters,fit.loglik)
            return run
        problem = SpatialPanelProblem(x,y,prepare_spatial_panel_weights(w,period_ids=np.arange(t)))
        if kkp:
            from statadic.inference.linear_covariance import classical_covariance
            estimator = KKP(full_weights=case.endswith('_full'),tolerance=1e-9)
            def run():
                fit = estimator.estimate(problem)
                return pack(fit.parameters,fit.spatial_lambda,fit.variance_components,classical_covariance(fit.linear).covariance.values)
            return run
        import jax.numpy as jnp
        estimator = PanelML(SAR() if sar else SEM())
        def run():
            fit = estimator.estimate(problem)
            return pack(fit.parameters,fit.loglik,jnp.linalg.inv(fit.information)[:-1,:-1])
        return run
    import spreg
    from libpysal.weights import full2W
    graph = full2W(w)
    if kkp:
        from functools import partial
        from unittest.mock import patch
        from scipy import optimize
        strict_lbfgsb = partial(optimize.fmin_l_bfgs_b,factr=10.0,pgtol=1e-12,maxiter=10000,maxfun=20000)
        def run():
            with patch.object(optimize,'fmin_l_bfgs_b',strict_lbfgsb):
                fit = spreg.GM_KKP(y.reshape(-1,1),x[:,:,1:].reshape(t*n,p-1),graph,full_weights=case.endswith('_full'))
            return pack(fit.betas,fit.vm)
        return run
    cls = spreg.Panel_FE_Lag if sar else spreg.Panel_FE_Error
    def run():
        design = np.concatenate([x,np.einsum('ij,tjp->tip',w,x)],axis=2) if observed else x
        fit = cls(y.reshape(-1,1),design.reshape(t*n,-1),graph,epsilon=1e-9)
        outputs = (fit.betas,fit.logll + (t*n)/2*np.log(t*n))
        return pack(*outputs) if observed else pack(*outputs,fit.vm)
    return run
