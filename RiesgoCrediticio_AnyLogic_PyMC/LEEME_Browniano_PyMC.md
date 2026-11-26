# RiesgoCrediticio — Modelo AnyLogic + movimiento browniano + PyMC

## Qué encontrarás

- `RiesgoCrediticio_Browniano_PyMC.alp`: prototipo AnyLogic 8.9.x, en XML `.alp`.
- `clientes_parametros.csv`: 100 clientes con variables y **probabilidades sintéticas para la demostración**.
- `coeficientes_pymc.csv`: coeficientes **hipotéticos** para la demostración, incluida `gamma`.
- `ajustar_pymc.py`: script para **estimar** parámetros bayesianos con un historial de clientes y producir ambos CSV.
- `requirements.txt`: versiones recomendadas para un entorno virtual de Python separado.

**AVISO SOBRE VALIDACIÓN**: Se verificó la estructura XML (sin errores de parseo, IDs únicos, 100 círculos vinculados a índices de clientes, dos gráficas, tres transiciones, un evento OU) y la sintaxis del script Python. No se dispone del ejecutable AnyLogic 8.9.10 en este entorno y NO se pudo compilar ni ejecutar el modelo en AnyLogic. Podría requerir ajustes mediante el editor antes de presentar el proyecto.

## 1. Iniciar el modelo en AnyLogic

1. Extrae *toda* la carpeta ZIP (no abras el `.alp` desde dentro del ZIP).
2. Conserva el `.alp` y ambos CSV **en la misma carpeta**.
3. Abre AnyLogic de escritorio → `File > Open` → `RiesgoCrediticio_Browniano_PyMC.alp`.
4. Si AnyLogic propone convertir el archivo al formato de tu versión, acepta y conserva una copia del original.
5. En `Projects`, selecciona el experimento `Simulation` y ejecuta el modelo. Verifica el panel `Problems`.
6. En `Main`, la rejilla de 10×10 representa a los 100 clientes. Verde=normal, amarillo=mora, rojo=incumplimiento.
7. Dos Time Plot muestran el número de clientes por estado y el factor browniano `estresEconomico`.

**Si los CSV no se leen**: AnyLogic resuelve rutas relativas desde el directorio de trabajo de ejecución, que puede no coincidir con el del `.alp`. Abre `Main > Startup code` y actualiza los nombres `clientes_parametros.csv` y `coeficientes_pymc.csv` a sus rutas absolutas. Cuando no los encuentra, el modelo usa parámetros sintéticos internos y comunica esa condición en la consola.

## 2. Matemática usada en AnyLogic

- Estado inicial: `normal` para 100 clientes.
- Tasa Normal→Mora: 0.08 por mes.
- Tasa Mora→Normal: 0.15 por mes.
- Tasa Mora→Incumplimiento: individual y dependiente del tiempo.
- OU con Euler–Maruyama cada 0.25 meses:

  `X <- X + kappa*(mu-X)*0.25 + sigma*sqrt(0.25)*Z`, `Z ~ N(0,1)`.

- `pBase_i` se define como la PD **mensual** del cliente en mora, en el escenario `X=0`, en ausencia de la transición competidora de recuperación. Para incorporar el factor macroeconómico preservando exactamente la regresión logística de PyMC:

  `logit(p_i(t)) = logit(pBase_i) + gamma * X_t`.

- Se pasa de esa probabilidad a la tasa instantánea tipo Rate:

  `lambda_i(t) = -log(1-p_i(t)) = log(1+exp(logit(pBase_i)+gamma*X_t))`.

La transición Mora→Normal con tasa 0.15 compite contra Mora→Default. Por ello, la **probabilidad efectiva** de default durante un mes con recuperaciones activas no es simplemente `p_i(t)`; el modelo simula dicha competencia.

En cada paso de OU, `Main` llama `c.onChange()` sobre los clientes, para que AnyLogic reprograme las transiciones con disparador `Rate`.

## 3. Estimación bayesiana en PyMC

Abre una terminal en la misma carpeta, usa preferentemente un entorno virtual e instala:

```bash
python -m pip install -r requirements.txt
python ajustar_pymc.py
```

**Datos propios:** guarda `historico_mora.csv` con encabezados EXACTOS:

```
ingreso_z,endeudamiento,estres,default_siguiente_mes
```

Cada fila debe ser una observación cliente-mes cuando el crédito estaba en **mora** al inicio del periodo. `default_siguiente_mes` vale 0 o 1. La variable `estres` debe estar alineada temporalmente y no usar información del futuro. `ingreso_z` debe estar transformado con media/desviación calculada solo en los datos de entrenamiento. `endeudamiento` aquí representa una razón numérica estandarizada por criterio explícito (en los datos demo está entre 0.05 y 0.95).

Si no existe el archivo histórico, el script genera un conjunto sintético (para enseñanza). En tal caso, las conclusiones *no son inferencia real del riesgo bancario*. El prototipo utiliza la regresión logística bayesiana:

`logit(P(default=1 | mora, x)) = beta0 + beta_ingreso*ingreso_z + beta_deuda*endeudamiento + gamma*estres`.

Posteriormente genera `clientes_parametros.csv` y `coeficientes_pymc.csv` que AnyLogic lee en `Main > Startup code`.

**Limitación estadística actual:** en el modelo de PyMC, `default_siguiente_mes` es Bernoulli y no se ha especificado todavía la probabilidad competidora de `Mora→Normal`. Para una estimación empírica correcta de las tasas de transición en tiempo continuo, el siguiente desarrollo debe usar un modelo de riesgos competitivos (default/recuperación) con historia temporal. La regresión Bernoulli sirve como aproximación inicial/prototipo.

**Limitación bayesiana actual:** AnyLogic recibe medias posteriores, no muestras completas de todos los coeficientes. Para propagar la *incertidumbre epistemológica*, en la siguiente etapa repetiremos simulaciones asignando una muestra posterior conjunta de `(beta0,beta_ingreso,beta_deuda,gamma)` por escenario y recalculando `pBase` para los 100 clientes.

## 4. Verificaciones de calidad

1. Al iniciar: `Normal=100`, `Mora=0`, `Default=0`.
2. Siempre: `Normal + Mora + Default = 100`.
3. El número en Default no disminuye (estado absorbente).
4. Al subir `sigma`, la dispersión de `X_t` entre repeticiones debe aumentar.
5. Al usar `gamma=0`, la trayectoria macroeconómica no debe influir en los defaults.
6. Cuando se importen datos reales, los coeficientes deben contrastarse con pruebas predictivas fuera de muestra y diagnósticos de MCMC (R-hat, ESS, divergencias).

## 5. Problemas conocidos / errores comunes

- **No aparecen líneas**: comprueba el tiempo simulado y que las dos Time Plot estén actualizadas automáticamente.
- **No aparecen los 100 puntos**: comprueba que `Main` tenga la población `clientes` con 100 agentes. Los puntos se dibujan como 100 óvalos dinámicos de `Main`, cada uno enlazado a `clientes.get(i)`, no son 100 dibujos de formas dentro de `cliente`.
- **`get_Main()` sin resolver**: selecciona el agente `cliente` y reemplaza `get_Main()` por la referencia al agente superior que ofrezca el autocompletado de tu versión; también es válida la referencia `main` si existe como puntero de propietario generado por AnyLogic.
- **Error al abrir ALP**: la estructura es un prototipo generado y no compilado. Comparte el primer mensaje de `Problems` para adaptarla a la versión exacta de escritorio.
- **La instalación Python falla por ArviZ**: usa un entorno virtual aislado y las versiones indicadas en `requirements.txt`. Esta ejecución no se comprobó aquí porque el entorno de pruebas combina PyMC 5.27.1 con ArviZ 1.1.0, que son incompatibles en `arviz.concat`.

## 6. Qué presentar como avance

- Arquitectura PyMC→CSV→AnyLogic.
- Estado Normal/Mora/Default para cada cliente.
- Factor OU representando un choque económico compartido.
- Animación de colores de 100 indicadores individuales.
- Dos gráficas en tiempo real.
- Resultados y limitaciones documentados, distinguiendo datos sintéticos de estimaciones empíricas.
