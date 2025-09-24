# vbt_extensions/metrics/riskfolio_adapter.py
import riskfolio as rl


def riskfolio_meanvar(returns):
    port = rl.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="hist")
    w = port.optimization(model="Classic", obj="Sharpe")
    return w
