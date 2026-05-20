"""Spanish (Castilian) user-facing strings, centralized for easy auditing."""

# Auth
AUTH_TITLE = "Acceso al simulador"
AUTH_PROMPT = "Introduce la contraseña"
AUTH_BUTTON = "Entrar"
AUTH_ERROR = "Contraseña incorrecta."
AUTH_MISSING_SECRET = (
    "No se ha configurado la contraseña de la aplicación. "
    "Crea .streamlit/secrets.toml con la clave app_password."
)

# App
APP_TITLE = "Simulador DCA: Puro vs Táctico"
APP_SUBTITLE = (
    "Compara una estrategia DCA pura frente a una DCA táctica sobre un mismo "
    "activo, usando datos históricos y simulaciones de Montecarlo."
)

# Sidebar
SIDEBAR_HEADER = "Parámetros de la simulación"
INPUT_NAME = "Nombre de la inversión"
INPUT_TICKER = "Ticker o ISIN"
INPUT_INITIAL = "Inversión inicial (€)"
INPUT_MONTHLY = "Inversión mensual (€)"
INPUT_YEARS_HIST = "Años de histórico a usar"
INPUT_YEARS_MC = "Años a simular en el Montecarlo"
RUN_BUTTON = "Ejecutar simulación"

# Disclaimer
DISCLAIMER = (
    "Aviso: esta herramienta tiene fines educativos. No constituye "
    "asesoramiento financiero. Los rendimientos pasados no garantizan "
    "rendimientos futuros."
)

# Errors / status
ERR_NO_DATA = (
    "No se han encontrado datos para el ticker indicado. "
    "Comprueba el símbolo en Yahoo Finance."
)
ERR_ISIN_UNRESOLVED = (
    "No se ha podido resolver el ISIN automáticamente. "
    "Por favor, introduce un ticker de Yahoo Finance."
)
ERR_INSUFFICIENT_HISTORY = (
    "El histórico disponible es más corto que el solicitado. "
    "Reduce los años de histórico o usa otro ticker."
)
PROGRESS_MC = "Ejecutando simulaciones..."

# Tabs
TAB_HISTORICAL = "Histórico"
TAB_MONTECARLO = "Montecarlo"

# Historical tab
HIST_SUMMARY_TITLE = "Resumen del backtest histórico"
HIST_COL_STRATEGY = "Estrategia"
HIST_COL_INVESTED = "Total invertido (€)"
HIST_COL_COMMITTED = "Capital comprometido (€)"
HIST_COL_FINAL = "Valor final (€)"
HIST_COL_TOTAL_RETURN = "Rendimiento total"
HIST_COL_CAGR = "CAGR (TIR anualizada)"
HIST_COL_TWR = "TWR anualizado"
HIST_COL_VOL = "Volatilidad anualizada"
HIST_COL_SHARPE = "Ratio de Sharpe"
HIST_COL_MDD = "Máximo drawdown"
HIST_SHARPE_NOTE = "Nota: ratio de Sharpe calculado con tasa libre de riesgo = 0."
HIST_CHART_PORTFOLIO = "Valor de la cartera a lo largo del tiempo"
HIST_CHART_PRICE = "Precio del activo"
HIST_LEGEND_INVESTED = "Total invertido"

# Strategy labels
LABEL_PURO = "DCA Puro"
LABEL_TACTICO = "DCA Táctico"

# Monte Carlo tab
MC_TARGET_INPUT = "Beneficio objetivo X (€)"
MC_TARGET_CAPTION = (
    "Al cambiar X solo se actualiza la probabilidad P(beneficio > X) y la línea "
    "vertical de la CDF. No se vuelve a ejecutar la simulación."
)
MC_STATS_TITLE = "Estadísticos del valor final"
MC_STAT_MEAN = "Media"
MC_STAT_MEDIAN = "Mediana"
MC_PROBS_TITLE = "Probabilidades"
MC_PROB_RUIN = "P(quiebra técnica) — valor final < 1 €"
MC_PROB_NEG = "P(en negativo) — valor final < total invertido"
MC_PROB_TARGET = "P(beneficio > X)"
MC_FAN_CHART = "Recorrido de la cartera (percentiles 10/30/50/70/90)"
MC_HIST_CHART = "Histograma del valor final"
MC_PDF_CHART = "Densidad (KDE) del valor final"
MC_CDF_CHART = "CDF del valor final"
MC_COMPARE_HIST = "Comparativa histogramas: Puro vs Táctico"
MC_COMPARE_CDF = "Comparativa CDF: Puro vs Táctico"
MC_COMPARE_HIST_RETURN = "Comparativa histogramas — rendimiento total (%)"
MC_COMPARE_CDF_RETURN = "Comparativa CDF — rendimiento total (%)"
MC_SPAGHETTI = "Trayectorias simuladas (10.000 caminos)"

# Axis labels
AXIS_DATE = "Fecha"
AXIS_MONTH = "Mes"
AXIS_VALUE_EUR = "Valor (€)"
AXIS_PRICE_EUR = "Precio (€)"
AXIS_FREQ = "Frecuencia"
AXIS_DENSITY = "Densidad"
AXIS_PROB = "Probabilidad acumulada"
AXIS_RETURN_PCT = "Rendimiento total (%)"
