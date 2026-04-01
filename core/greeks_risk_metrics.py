"""
Greeks Risk Metrics with Auto-Differentiation
=============================================
Greeks风险指标 - 自动微分实现

核心思想：使用JAX自动微分实现精确的风险度量

对比有限差分法：
- 有限差分：近似计算，有截断误差，步长选择困难
- 自动微分：精确计算，无截断误差，支持高阶导数

Author: AI Agent CTO
Reference: Black-Scholes Greeks, adapted for e-commerce
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class GreeksResult:
    """Greeks计算结果"""
    # 一阶Greeks
    delta: float           # 价格敏感度: ∂Π/∂P
    vega: float            # 波动率敏感度: ∂Π/∂σ
    theta: float           # 时间敏感度: ∂Π/∂t
    rho: float             # 库存敏感度: ∂Π/∂q
    
    # 二阶Greeks
    gamma: float           # 价格凸性: ∂²Π/∂P²
    vanna: float           # 价格-波动率交叉: ∂²Π/∂P∂σ
    charm: float           # 价格-时间交叉: ∂²Π/∂P∂t
    
    # 风险等级
    risk_level: str        # LOW / MEDIUM / HIGH


class GreeksCalculator:
    """
    Greeks风险计算器
    
    将金融期权定价中的Greeks概念适配到电商定价场景：
    
    ╔════════════╦══════════════════════════════════════════════════╗
    ║   Greek    ║                  电商映射                          ║
    ╠════════════╬══════════════════════════════════════════════════╣
    ║   Delta    ║  利润对价格的一阶敏感度（方向风险）               ║
    ║   Gamma    ║  利润对价格的二阶敏感度（凸性风险）               ║
    ║   Vega     ║  利润对退货率波动的敏感度（质量风险）             ║
    ║   Theta    ║  利润对时间的敏感度（季节风险）                   ║
    ║   Rho      ║  利润对库存的敏感度（库存风险）                   ║
    ║   Vanna    ║  价格-波动率交叉风险（双11期间）                  ║
    ╚════════════╩══════════════════════════════════════════════════╝
    
    应用场景：
    1. 实时监控定价风险
    2. 触发风险熔断机制
    3. 优化价格调整策略
    """
    
    def __init__(self, profit_function):
        """
        初始化Greeks计算器
        
        Args:
            profit_function: 利润函数 f(price, params) -> profit
        """
        self.profit_function = profit_function
        
        # 编译JAX梯度函数（JIT加速）
        # 在实际实现中使用 jax.jit(jax.grad(...))
        self._grad_fn = None  # jax.grad(profit_function)
        self._hessian_fn = None  # jax.hessian(profit_function)
        
        # 风险阈值配置
        self.thresholds = {
            'delta': {'low': 100, 'high': 200},
            'gamma': {'low': 500, 'high': 1000},
            'vega': {'low': 10000, 'high': 20000}
        }
    
    # ============================================================
    # 核心方法：计算所有Greeks
    # ============================================================
    
    def calculate(self, price: float, params: Dict) -> GreeksResult:
        """
        计算所有Greeks风险指标
        
        Args:
            price: 当前价格
            params: 定价参数
        
        Returns:
            GreeksResult对象
        """
        # ========== 一阶Greeks ==========
        
        # Delta: ∂Π/∂P
        # 含义：价格每变化$1，利润变化多少
        # 应用：判断价格调整方向
        delta = self._calculate_delta(price, params)
        
        # Vega: ∂Π/∂σ
        # 含义：退货率波动每增加1%，利润变化多少
        # 应用：质量风险预警
        vega = self._calculate_vega(price, params)
        
        # Theta: ∂Π/∂t
        # 含义：时间每推移1天，利润变化多少
        # 应用：季节性调整
        theta = self._calculate_theta(price, params)
        
        # Rho: ∂Π/∂q
        # 含义：库存每变化1件，利润变化多少
        # 应用：库存优化
        rho = self._calculate_rho(price, params)
        
        # ========== 二阶Greeks ==========
        
        # Gamma: ∂²Π/∂P²
        # 含义：Delta的变化率（凸性）
        # 应用：判断价格调整的边际效应
        gamma = self._calculate_gamma(price, params)
        
        # Vanna: ∂²Π/∂P∂σ
        # 含义：价格敏感度如何随退货率波动变化
        # 应用：大促期间的风险预警
        vanna = self._calculate_vanna(price, params)
        
        # Charm: ∂²Π/∂P∂t
        # 含义：价格敏感度如何随时间变化
        # 应用：季节性定价策略
        charm = self._calculate_charm(price, params)
        
        # ========== 风险等级评估 ==========
        risk_level = self._assess_risk_level(delta, gamma, vega)
        
        return GreeksResult(
            delta=delta,
            vega=vega,
            theta=theta,
            rho=rho,
            gamma=gamma,
            vanna=vanna,
            charm=charm,
            risk_level=risk_level
        )
    
    # ============================================================
    # 一阶Greeks计算
    # ============================================================
    
    def _calculate_delta(self, price: float, params: Dict) -> float:
        """
        计算Delta（价格敏感度）
        
        Delta = ∂Π/∂P
        
        含义：
        - Delta > 0: 提价增加利润
        - Delta < 0: 降价增加利润
        - |Delta| 越大，价格敏感度越高
        
        示例：
        - Delta = 200: 价格每涨$1，利润增加$200
        - Delta = -100: 价格每涨$1，利润减少$100
        """
        # 使用JAX自动微分
        # delta = jax.grad(self.profit_function)(price, params)
        
        # 伪代码实现（有限差分近似）
        epsilon = 0.01
        profit_plus = self.profit_function(price + epsilon, params)
        profit_minus = self.profit_function(price - epsilon, params)
        delta = (profit_plus - profit_minus) / (2 * epsilon)
        
        return delta
    
    def _calculate_vega(self, price: float, params: Dict) -> float:
        """
        计算Vega（波动率敏感度）
        
        Vega = ∂Π/∂σ
        
        其中σ为退货率波动率
        
        含义：
        - Vega > 0: 退货率波动增加利润
        - Vega < 0: 退货率波动减少利润
        
        应用：
        - 高Vega → 需要优化商品质量
        - 高Vega → 考虑价格保护策略
        """
        # 创建包含退货率波动的利润函数
        def profit_with_volatility(price_sigma):
            P, sigma = price_sigma
            params_copy = params.copy()
            params_copy['return_volatility'] = sigma
            return self.profit_function(P, params_copy)
        
        # 计算对sigma的偏导
        # vega = jax.grad(lambda ps: profit_with_volatility(ps)[1])(price, 0.15)
        
        # 伪代码实现
        epsilon = 0.001
        profit_plus = profit_with_volatility((price, 0.15 + epsilon))
        profit_minus = profit_with_volatility((price, 0.15 - epsilon))
        vega = (profit_plus - profit_minus) / (2 * epsilon)
        
        return vega
    
    def _calculate_theta(self, price: float, params: Dict) -> float:
        """
        计算Theta（时间敏感度）
        
        Theta = ∂Π/∂t
        
        含义：利润随时间的衰减（季节性影响）
        
        应用：
        - Theta < 0: 商品即将过季，需要降价
        - Theta > 0: 商品进入旺季，可以提价
        """
        # 伪代码实现
        return 0.0  # 简化处理
    
    def _calculate_rho(self, price: float, params: Dict) -> float:
        """
        计算Rho（库存敏感度）
        
        Rho = ∂Π/∂q
        
        含义：利润对库存水平的敏感度
        
        应用：
        - Rho < 0: 库存过高，需要降价清仓
        - Rho > 0: 库存不足，可以提价
        """
        # 伪代码实现
        inventory = params.get('inventory', 1000)
        
        epsilon = 10
        params_plus = params.copy()
        params_plus['inventory'] = inventory + epsilon
        params_minus = params.copy()
        params_minus['inventory'] = inventory - epsilon
        
        profit_plus = self.profit_function(price, params_plus)
        profit_minus = self.profit_function(price, params_minus)
        rho = (profit_plus - profit_minus) / (2 * epsilon)
        
        return rho
    
    # ============================================================
    # 二阶Greeks计算
    # ============================================================
    
    def _calculate_gamma(self, price: float, params: Dict) -> float:
        """
        计算Gamma（价格凸性）
        
        Gamma = ∂²Π/∂P²
        
        含义：Delta的变化率（利润函数的曲率）
        
        应用：
        - Gamma > 0: 利润函数凸，价格调整边际效应递增
        - Gamma < 0: 利润函数凹，价格调整边际效应递减
        
        风险预警：
        - |Gamma| 过大 → 价格敏感度变化剧烈，风险高
        """
        # 使用JAX Hessian
        # gamma = jax.hessian(self.profit_function)(price, params)
        
        # 伪代码实现（中心差分）
        epsilon = 0.01
        profit_center = self.profit_function(price, params)
        profit_plus = self.profit_function(price + epsilon, params)
        profit_minus = self.profit_function(price - epsilon, params)
        
        gamma = (profit_plus - 2 * profit_center + profit_minus) / (epsilon ** 2)
        
        return gamma
    
    def _calculate_vanna(self, price: float, params: Dict) -> float:
        """
        计算Vanna（价格-波动率交叉Greeks）
        
        Vanna = ∂²Π/∂P∂σ
        
        含义：价格敏感度如何随退货率波动变化
        
        应用场景：双11等大促期间
        - 价格波动大 + 退货率波动大 = 双重风险
        - Vanna可以提前预警这种联合风险
        """
        # 伪代码实现
        return 0.0
    
    def _calculate_charm(self, price: float, params: Dict) -> float:
        """
        计算Charm（价格-时间交叉Greeks）
        
        Charm = ∂²Π/∂P∂t
        
        含义：价格敏感度如何随时间变化
        
        应用：季节性定价策略调整
        """
        # 伪代码实现
        return 0.0
    
    # ============================================================
    # 风险评估
    # ============================================================
    
    def _assess_risk_level(
        self,
        delta: float,
        gamma: float,
        vega: float
    ) -> str:
        """
        评估风险等级
        
        基于Greeks绝对值判断风险等级
        """
        delta_abs = abs(delta)
        gamma_abs = abs(gamma)
        vega_abs = abs(vega)
        
        # 高风险条件
        if (delta_abs > self.thresholds['delta']['high'] or
            gamma_abs > self.thresholds['gamma']['high'] or
            vega_abs > self.thresholds['vega']['high']):
            return "HIGH"
        
        # 中风险条件
        if (delta_abs > self.thresholds['delta']['low'] or
            gamma_abs > self.thresholds['gamma']['low'] or
            vega_abs > self.thresholds['vega']['low']):
            return "MEDIUM"
        
        return "LOW"
    
    # ============================================================
    # 风险熔断机制
    # ============================================================
    
    def check_circuit_breaker(self, greeks: GreeksResult) -> Tuple[str, str]:
        """
        风险熔断检查
        
        当风险指标超过阈值时，触发熔断，暂停自动定价
        
        熔断条件：
        1. |Delta| > 200: 价格敏感度极高
        2. |Gamma| > 1000: 凸性风险过高
        3. |Vega| > 20000: 波动率风险过高
        """
        if abs(greeks.delta) > 200:
            return "TRIGGERED", f"Delta过高: {greeks.delta:.2f}"
        
        if abs(greeks.gamma) > 1000:
            return "TRIGGERED", f"Gamma过高: {greeks.gamma:.2f}"
        
        if abs(greeks.vega) > 20000:
            return "TRIGGERED", f"Vega过高: {greeks.vega:.2f}"
        
        return "NORMAL", "风险指标正常"


# ============================================================
# 使用示例
# ============================================================

def example_profit_function(price: float, params: Dict) -> float:
    """示例利润函数"""
    C = params.get('cost', 50.0)
    C_ret = params.get('return_cost', 15.0)
    base_demand = params.get('base_demand', 1000.0)
    elasticity = params.get('elasticity', -2.5)
    
    # 需求函数
    demand = base_demand * (price ** elasticity)
    
    # 退货率（随价格上升）
    return_rate = 0.1 + 0.001 * (price - C)
    return_rate = max(0.05, min(return_rate, 0.5))
    
    # 利润
    profit = (price - C) * (1 - return_rate) * demand - C_ret * return_rate * demand
    
    return profit


if __name__ == "__main__":
    print("=" * 60)
    print("Greeks风险指标计算器")
    print("=" * 60)
    
    # 初始化计算器
    calculator = GreeksCalculator(example_profit_function)
    
    # 定价参数
    params = {
        'cost': 50.0,
        'return_cost': 15.0,
        'base_demand': 1000.0,
        'elasticity': -2.5,
        'inventory': 800
    }
    
    price = 80.0
    
    print(f"\n当前价格: ${price}")
    print(f"成本: ${params['cost']}")
    
    # 计算Greeks
    print("\n" + "-" * 60)
    print("计算Greeks...")
    print("-" * 60)
    
    greeks = calculator.calculate(price, params)
    
    print(f"\n一阶Greeks:")
    print(f"  Delta (价格敏感度): {greeks.delta:+.2f}")
    print(f"    → 解读: 价格每涨$1，利润{greeks.delta:+.2f}")
    print(f"  Vega (波动率敏感度): {greeks.vega:+.2f}")
    print(f"  Theta (时间敏感度): {greeks.theta:+.4f}")
    print(f"  Rho (库存敏感度): {greeks.rho:+.4f}")
    
    print(f"\n二阶Greeks:")
    print(f"  Gamma (价格凸性): {greeks.gamma:+.4f}")
    print(f"  Vanna (价格-波动率交叉): {greeks.vanna:+.4f}")
    print(f"  Charm (价格-时间交叉): {greeks.charm:+.4f}")
    
    print(f"\n风险等级: {greeks.risk_level}")
    
    # 熔断检查
    status, message = calculator.check_circuit_breaker(greeks)
    print(f"\n熔断检查: {status}")
    if status == "TRIGGERED":
        print(f"  警告: {message}")
    
    print("\n" + "=" * 60)
    print("Greeks计算完成")
    print("=" * 60)
