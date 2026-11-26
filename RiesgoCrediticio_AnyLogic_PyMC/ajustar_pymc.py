"""
RiesgoCrediticio - Estimacion bayesiana del incumplimiento mensual condicional a MORA.

Entradas opcionales en la misma carpeta:
  historico_mora.csv: ingreso_z,endeudamiento,estres,default_siguiente_mes
  clientes_simulacion.csv: ingreso_z,endeudamiento (100 filas)

Salidas:
  coeficientes_pymc.csv: parametro,valor
  clientes_parametros.csv: id,ingreso_z,endeudamiento,p_mensual_mora

ADVERTENCIA: si falta historico_mora.csv, se generan datos sintéticos demostrativos;
no son estimaciones reales de un banco.

Ejecucion: pip install pymc arviz pandas numpy
           python ajustar_pymc.pyd
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm

SEED = 20261125
RNG = np.random.default_rng(SEED)
HERE = Path(__file__).resolve().parent
HIST = HERE / 'historico_mora.csv'
CLIENTS = HERE / 'clientes_simulacion.csv'

def logistic(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))

if HIST.exists():
    df = pd.read_csv(HIST)
    print('Usando historial real proporcionado por el usuario:', HIST)
else:
    n = 1800
    ingreso = RNG.normal(size=n)
    deuda = RNG.uniform(0.05,0.95,size=n)
    estres = RNG.normal(0,0.65,size=n)
    logit_p = -2.7 -0.5*ingreso+1.4*deuda+0.60*estres
    y = RNG.binomial(1,logistic(logit_p))
    df = pd.DataFrame({'ingreso_z':ingreso,'endeudamiento':deuda,
                       'estres':estres,'default_siguiente_mes':y})
    df.to_csv(HERE/'historico_sintetico_ejemplo.csv', index=False)
    print('ATENCION: el historial es SINTETICO. No interpretes coeficientes como evidencia real.')

cols=['ingreso_z','endeudamiento','estres','default_siguiente_mes']
if any(c not in df.columns for c in cols):
    raise ValueError(f'Faltan columnas. Se esperan: {cols}')
df=df[cols].dropna().copy()
if not df.default_siguiente_mes.isin([0,1]).all():
    raise ValueError('default_siguiente_mes debe valer 0 o 1')

with pm.Model() as model:
    beta0 = pm.Normal('beta0', mu=-2.5, sigma=1.5)
    beta_ingreso = pm.Normal('beta_ingreso', mu=0, sigma=1)
    beta_deuda = pm.Normal('beta_deuda', mu=0, sigma=1)
    gamma = pm.HalfNormal('gamma', sigma=1)
    logit_pd = (beta0 + beta_ingreso * df.ingreso_z.to_numpy() +
                beta_deuda * df.endeudamiento.to_numpy() +
                gamma * df.estres.to_numpy())
    pm.Bernoulli('y_observada', logit_p=logit_pd,
                 observed=df.default_siguiente_mes.to_numpy(dtype=int))
    idata = pm.sample(draws=400, tune=400, chains=2, cores=1,
                      target_accept=0.90, random_seed=SEED, progressbar=True)

names=['beta0','beta_ingreso','beta_deuda','gamma']
coef={name:float(idata.posterior[name].mean()) for name in names}
pd.DataFrame([{'parametro':name,'valor':coef[name]} for name in names]).to_csv(
    HERE/'coeficientes_pymc.csv',index=False)
print('Medias posteriores:',coef)

if CLIENTS.exists():
    cli=pd.read_csv(CLIENTS)[['ingreso_z','endeudamiento']]
    if len(cli)!=100:raise ValueError('clientes_simulacion.csv debe tener 100 filas.')
else:
    cli=pd.DataFrame({'ingreso_z':RNG.normal(size=100),
                      'endeudamiento':RNG.uniform(0.05,0.95,size=100)})
    cli.to_csv(HERE/'clientes_simulacion.csv',index=False)

# Prediccion basica de PD en estres=0, aplicable SOLO al estado de mora.
eta=(coef['beta0']+coef['beta_ingreso']*cli.ingreso_z.to_numpy()+
     coef['beta_deuda']*cli.endeudamiento.to_numpy())
resultado=pd.DataFrame({'id':np.arange(100),
                        'ingreso_z':cli.ingreso_z.to_numpy(),
                        'endeudamiento':cli.endeudamiento.to_numpy(),
                        'p_mensual_mora':logistic(eta)})
resultado.to_csv(HERE/'clientes_parametros.csv',index=False)
print('CSV creados: clientes_parametros.csv y coeficientes_pymc.csv')
print('PD promedio demo:', resultado.p_mensual_mora.mean())
print('Importante: se exporta una media posterior (estimacion puntual) en esta version.')
