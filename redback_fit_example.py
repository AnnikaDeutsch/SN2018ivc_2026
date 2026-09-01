import redback
import pandas as pd
import numpy as np

data = pd.read_csv('data/radio_18ivc_data.csv')

# redback expects frequency in Hz
transient = redback.transient.Transient(
    name='SN2018ivc',
    data_mode='flux_density',
    time=data['phase'].values,           # days
    flux_density=data['flux'].values,    # mJy
    flux_density_err=data['flux_err'].values,
    frequency=data['freq'].values * 1e9  # GHz → Hz
)

model = 'synchrotron_massloss'
priors = redback.priors.get_priors(model=model)
priors['redshift'] = 0.003793  # NGC 1068, fix it

result = redback.fit_model(
    transient=transient, model=model,
    sampler='dynesty', model_kwargs=dict(frequency=data['freq'].values * 1e9),
    prior=priors, nlive=500
)
result.plot_corner()
result.plot_multiband_lightcurve()
