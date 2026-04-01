"""
Market Maker Pricing Model - Pseudocode
=======================================
做市商定价格模型 - 伪代码实现

核心思想：将金融做市商模型迁移到电商定价场景

Author: AI Agent CTO
Reference: Avellaneda & Stoikov (2008)
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class PricingParams:
    """定价参数"""
    cost: float              # 采购成本 C
    return_cost: float       # 退货成本 C_ret
    target_margin: float     # 目标利润率


class MarketMakerPricingEngine:
    """
    做市商定价格引擎
    
    核心优化方程：
        max_P E[Π] = (P - C) · (1 - R(P)) · Q(P) - C_ret · R(P) · Q(P) - H(q)
    
    其中：
    - P: 动态售价
    - C: 采购成本
    - Q(P): 销量函数（需求曲线）
    - R(P): 退货率函数（价格的函数）
    - C_ret: 退货净损失
    - H(q): 库存持有成本
    """
    
    def __init__(self, params: PricingParams, elasticity: float = -2.5):
        self.params = params
        self.elasticity = elasticity
    
    # ============================================================
    # 核心模型：需求函数
    # ============================================================
    
    def demand_function(self, price: float, base_demand: float) -> float:
        """
        需求函数（对数线性模型）
        
        Q(P) = Q_0 · P^ε
        
        其中 ε 为价格弹性（通常为负值）
        
        直觉：价格越高，需求越低
        """
        return base_demand * (price ** self.elasticity)
    
    # ============================================================
    # 核心模型：退货率函数
    # ============================================================
    
    def return_rate_function(self, price: float, base_rate: float) -> float:
        """
        退货率函数（Logistic模型）
        
        R(P) = R_min + (R_max - R_min) / (1 + exp(-k · (P - P_mid)))
        
        关键洞察：价格越高，用户期望越高，退货率随之上升
        
        例如：
        - $50的商品，退货率10%
        - $100的同类型商品，退货率可能升至20%
        """
        R_min = base_rate * 0.5
        R_max = min(base_rate * 2.0, 0.5)  # 上限50%
        P_mid = self.params.cost * 1.5
        k = 0.01
        
        return R_min + (R_max - R_min) / (1 + exp(-k * (price - P_mid)))
    
    # ============================================================
    # 核心模型：利润函数
    # ============================================================
    
    def profit_function(
        self,
        price: float,
        base_demand: float,
        base_return_rate: float,
        inventory: int
    ) -> float:
        """
        利润函数
        
        E[Π] = (P - C) · (1 - R(P)) · Q(P) - C_ret · R(P) · Q(P) - H(q)
        
        分解：
        1. (P - C) · (1 - R(P)) · Q(P): 净销售收入
        2. C_ret · R(P) · Q(P): 退货成本
        3. H(q): 库存持有成本
        """
        Q = self.demand_function(price, base_demand)
        R = self.return_rate_function(price, base_return_rate)
        
        # 净销售收入
        net_revenue = (price - self.params.cost) * (1 - R) * Q
        
        # 退货成本
        return_cost = self.params.return_cost * R * Q
        
        # 库存成本
        inventory_cost = self._inventory_cost(inventory)
        
        return net_revenue - return_cost - inventory_cost
    
    # ============================================================
    # 核心算法：最优价格（解析解）
    # ============================================================
    
    def optimal_price(
        self,
        base_demand: float,
        base_return_rate: float,
        inventory: int,
        price_bounds: Tuple[float, float] = None
    ) -> Dict:
        """
        计算最优价格（解析解）
        
        P* = (|ε| / (|ε| - 1)) · (C + (R · C_ret) / (1 - R))
        
        公式解读：
        - 第一项 (|ε| / (|ε| - 1)): 成本加成系数（Lerner指数变形）
        - 第二项 (C + (R · C_ret) / (1 - R)): 风险调整后的有效成本
        
        关键洞察：
        1. 退货率R越高，最优价格P*越高（需要更多风险溢价）
        2. 价格弹性|ε|越大，定价能力越弱（竞争激烈）
        3. 退货成本C_ret越高，风险溢价越大
        """
        C = self.params.cost
        C_ret = self.params.return_cost
        epsilon = abs(self.elasticity)
        
        # 风险调整后的有效成本
        # 包含退货风险的预期成本
        effective_cost = C + (base_return_rate * C_ret) / (1 - base_return_rate + 1e-6)
        
        # 成本加成系数
        markup = epsilon / (epsilon - 1)
        
        # 最优价格
        optimal_price = markup * effective_cost
        
        # 应用价格约束
        if price_bounds:
            min_price, max_price = price_bounds
            optimal_price = max(min_price, min(optimal_price, max_price))
        
        # 计算预期利润
        expected_profit = self.profit_function(
            optimal_price, base_demand, base_return_rate, inventory
        )
        
        return {
            'optimal_price': optimal_price,
            'effective_cost': effective_cost,
            'markup': markup,
            'expected_profit': expected_profit,
            'expected_margin': (optimal_price - C) / optimal_price
        }
    
    # ============================================================
    # 辅助函数：库存成本
    # ============================================================
    
    def _inventory_cost(self, inventory: int, target: int = 1000) -> float:
        """
        非对称库存成本函数
        
        电商库存特点：
        - 过剩库存：资金占用 + 仓储 + 贬值
        - 缺货库存：销售损失 + 排名下降
        
        H(q) = {
            h⁺ · ((q - q_target) / q_target)²    if q > q_target
            h⁻ · ((q_safe - q) / q_safe)²        if q < q_safe
            h_base · |q - q_target| / q_target   otherwise
        }
        """
        h_plus = 0.05   # 过剩成本系数
        h_minus = 0.15  # 缺货成本系数
        q_safe = int(target * 0.3)
        
        if inventory > target:
            return h_plus * ((inventory - target) / target) ** 2
        elif inventory < q_safe:
            return h_minus * ((q_safe - inventory) / q_safe) ** 2
        else:
            return 0.02 * abs(inventory - target) / target


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    # 场景：电子产品定价
    params = PricingParams(
        cost=50.0,          # 成本$50
        return_cost=15.0,   # 退货成本$15
        target_margin=0.20  # 目标利润率20%
    )
    
    engine = MarketMakerPricingEngine(params, elasticity=-2.5)
    
    # 计算最优价格
    result = engine.optimal_price(
        base_demand=1000,
        base_return_rate=0.15,
        inventory=800,
        price_bounds=(55, 100)
    )
    
    print("=" * 50)
    print("做市商定价格引擎 - 定价建议")
    print("=" * 50)
    print(f"最优价格: ${result['optimal_price']:.2f}")
    print(f"有效成本: ${result['effective_cost']:.2f}")
    print(f"成本加成: {result['markup']:.2f}x")
    print(f"预期利润率: {result['expected_margin']*100:.1f}%")
    print(f"预期利润: ${result['expected_profit']:.2f}")
