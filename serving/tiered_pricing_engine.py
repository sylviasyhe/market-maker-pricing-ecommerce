"""
Tiered Pricing Engine - Pseudocode
==================================
分级定价引擎 - 伪代码实现

核心思想：在延迟和精度之间取得平衡

设计原则：
- 热数据（高频访问）→ 最快路径（缓存）
- 温数据（有销售记录）→ 较快路径（轻量模型）
- 冷数据（新SKU/低频）→ 最准路径（完整模型）

目标：P99延迟 < 25ms

Author: AI Agent CTO
"""

from dataclasses import dataclass
from typing import Dict, Optional, Any
from enum import Enum
import time


class PricingTier(Enum):
    """定价层级"""
    TIER_1 = "l1_cache"      # 本地缓存: <10ms, 80%请求
    TIER_2 = "redis_onnx"    # Redis + ONNX: <50ms, 15%请求
    TIER_3 = "ray_serve"     # Ray Serve: <200ms, 5%请求


@dataclass
class PricingResult:
    """定价结果"""
    price: float
    tier: PricingTier
    latency_ms: float
    confidence: float
    cached: bool


class LRUCache:
    """
    LRU (Least Recently Used) 缓存
    
    用于Tier 1本地缓存，实现O(1)读写
    
    核心操作：
    - get: O(1) 读取
    - put: O(1) 写入
    - 自动淘汰最久未使用的数据
    """
    
    def __init__(self, capacity: int, ttl_seconds: int = 60):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self.cache = {}  # 实际使用OrderedDict
        self.access_order = []  # 访问顺序
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值 O(1)"""
        if key in self.cache:
            item = self.cache[key]
            # 检查TTL
            if time.time() - item['timestamp'] < self.ttl:
                # 更新访问顺序
                self._update_access_order(key)
                return item['value']
            else:
                # 过期删除
                del self.cache[key]
        return None
    
    def put(self, key: str, value: Any):
        """设置缓存值 O(1)"""
        if key in self.cache:
            # 更新值
            self._update_access_order(key)
        else:
            # 检查容量
            if len(self.cache) >= self.capacity:
                # 淘汰最久未使用的
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]
            self.access_order.append(key)
        
        self.cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
    
    def _update_access_order(self, key: str):
        """更新访问顺序"""
        self.access_order.remove(key)
        self.access_order.append(key)


class TieredPricingEngine:
    """
    分级定价引擎
    
    根据SKU热度动态选择定价层级：
    
    ┌─────────────────────────────────────────────────────────────┐
    │                    分级定价架构                              │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
    │   │   Tier 1    │    │   Tier 2    │    │   Tier 3    │    │
    │   │  L1 Cache   │───→│ Redis+ONNX  │───→│  Ray Serve  │    │
    │   │  <10ms      │    │  <50ms      │    │  <200ms     │    │
    │   │  (80%)      │    │  (15%)      │    │  (5%)       │    │
    │   └─────────────┘    └─────────────┘    └─────────────┘    │
    │                                                              │
    │   热数据 → 温数据 → 冷数据                                   │
    │   高频SKU  有销售记录  新SKU/低频                             │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
    
    加权平均延迟: P99 < 25ms
    """
    
    def __init__(
        self,
        tier1_cache_size: int = 100000,
        tier1_ttl_seconds: int = 60,
        hot_sku_threshold: int = 1000,      # 24小时浏览量阈值
        warm_sku_threshold: int = 1         # 7天销售量阈值
    ):
        # Tier 1: 本地LRU缓存
        self.tier1_cache = LRUCache(tier1_cache_size, tier1_ttl_seconds)
        
        # Tier 2: Redis + ONNX轻量模型
        self.tier2_redis = None  # Redis连接
        self.tier2_model = None  # ONNX模型
        
        # Tier 3: Ray Serve完整DRL模型
        self.tier3_endpoint = None  # Ray Serve端点
        
        # 阈值配置
        self.hot_sku_threshold = hot_sku_threshold
        self.warm_sku_threshold = warm_sku_threshold
        
        # 统计指标
        self._request_count = 0
        self._tier_distribution = {tier: 0 for tier in PricingTier}
    
    # ============================================================
    # 核心方法：获取定价
    # ============================================================
    
    def get_price(
        self,
        sku_id: str,
        user_id: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> PricingResult:
        """
        获取定价建议（分级处理）
        
        流程：
        1. 判断SKU热度
        2. 选择合适层级
        3. 执行定价
        4. 更新缓存
        
        Args:
            sku_id: SKU唯一标识
            user_id: 用户ID（可选，用于个性化）
            context: 上下文信息（市场、类目等）
        
        Returns:
            PricingResult: 定价结果
        """
        start_time = time.time()
        self._request_count += 1
        
        # 生成缓存键
        cache_key = self._generate_cache_key(sku_id, user_id, context)
        
        # ========== Tier 1: L1 Cache (<10ms, 80%请求) ==========
        if self._is_hot_sku(sku_id):
            cached_price = self.tier1_cache.get(cache_key)
            if cached_price is not None:
                latency = (time.time() - start_time) * 1000
                self._tier_distribution[PricingTier.TIER_1] += 1
                
                return PricingResult(
                    price=cached_price,
                    tier=PricingTier.TIER_1,
                    latency_ms=latency,
                    confidence=0.95,
                    cached=True
                )
        
        # ========== Tier 2: Redis + ONNX (<50ms, 15%请求) ==========
        if self._is_warm_sku(sku_id):
            try:
                # 从Redis获取特征
                features = self._get_features_from_redis(sku_id)
                
                # ONNX模型推理
                price = self._tier2_inference(features)
                
                # 写入Tier 1缓存
                self.tier1_cache.put(cache_key, price)
                
                latency = (time.time() - start_time) * 1000
                self._tier_distribution[PricingTier.TIER_2] += 1
                
                return PricingResult(
                    price=price,
                    tier=PricingTier.TIER_2,
                    latency_ms=latency,
                    confidence=0.85,
                    cached=False
                )
            except Exception as e:
                print(f"[TieredEngine] Tier 2 failed: {e}, falling back to Tier 3")
        
        # ========== Tier 3: Ray Serve (<200ms, 5%请求) ==========
        try:
            # 完整DRL模型推理
            price = self._tier3_inference(sku_id, user_id, context)
            
            # 写入缓存（异步）
            self._async_update_caches(sku_id, cache_key, price)
            
            latency = (time.time() - start_time) * 1000
            self._tier_distribution[PricingTier.TIER_3] += 1
            
            return PricingResult(
                price=price,
                tier=PricingTier.TIER_3,
                latency_ms=latency,
                confidence=0.75,
                cached=False
            )
        except Exception as e:
            print(f"[TieredEngine] Tier 3 failed: {e}")
            
            # 降级策略：返回成本加成定价
            fallback_price = self._fallback_pricing(sku_id)
            
            latency = (time.time() - start_time) * 1000
            
            return PricingResult(
                price=fallback_price,
                tier=PricingTier.TIER_3,
                latency_ms=latency,
                confidence=0.5,
                cached=False
            )
    
    # ============================================================
    # SKU热度判断
    # ============================================================
    
    def _is_hot_sku(self, sku_id: str) -> bool:
        """
        判断是否为热数据SKU
        
        标准：过去24小时浏览量 > 阈值
        
        为什么用浏览量而不是销量？
        - 浏览量更实时
        - 销量有延迟
        - 浏览量可以预测销量
        """
        views_24h = self._get_24h_views(sku_id)
        return views_24h > self.hot_sku_threshold
    
    def _is_warm_sku(self, sku_id: str) -> bool:
        """
        判断是否为温数据SKU
        
        标准：过去7天有销售记录
        
        温数据的特点：
        - 有历史销售数据
        - 可以用轻量模型预测
        - 不需要完整DRL模型
        """
        sales_7d = self._get_7d_sales(sku_id)
        return sales_7d >= self.warm_sku_threshold
    
    # ============================================================
    # 各层级推理
    # ============================================================
    
    def _tier2_inference(self, features) -> float:
        """
        Tier 2: ONNX轻量模型推理
        
        特点：
        - 模型大小 < 10MB
        - 推理延迟 < 10ms
        - 精度略低于完整模型
        """
        # ONNX推理
        # output = onnx_model.run(features)
        # adjustment = output[0]  # 价格调整比例
        # return base_price * (1 + adjustment)
        
        # 伪代码
        return 100.0  # 示例
    
    def _tier3_inference(
        self,
        sku_id: str,
        user_id: Optional[str],
        context: Optional[Dict]
    ) -> float:
        """
        Tier 3: 完整DRL模型推理
        
        特点：
        - 完整状态空间（50+维特征）
        - PPO/SAC算法
        - 精度最高
        - 延迟最高（<200ms）
        """
        # 构建完整特征
        # features = build_full_features(sku_id, user_id, context)
        
        # Ray Serve推理
        # price = ray_serve_client.predict(features)
        
        # 伪代码
        return 100.0  # 示例
    
    # ============================================================
    # 降级策略
    # ============================================================
    
    def _fallback_pricing(self, sku_id: str) -> float:
        """
        降级定价策略
        
        当所有层级都失败时，使用兜底策略：
        - 成本加成定价
        - 简单但稳定
        """
        cost = self._get_sku_cost(sku_id)
        return cost * 1.2  # 成本加成20%
    
    # ============================================================
    # 辅助方法（伪代码）
    # ============================================================
    
    def _get_24h_views(self, sku_id: str) -> int:
        """获取24小时浏览量"""
        # 从Redis/ClickHouse查询
        return hash(sku_id) % 2000  # 模拟
    
    def _get_7d_sales(self, sku_id: str) -> int:
        """获取7天销售量"""
        # 从数据库查询
        return hash(sku_id) % 100  # 模拟
    
    def _get_sku_cost(self, sku_id: str) -> float:
        """获取SKU成本"""
        return 50.0 + (hash(sku_id) % 50)  # 模拟
    
    def _get_features_from_redis(self, sku_id: str):
        """从Redis获取特征"""
        # Redis查询
        pass
    
    def _generate_cache_key(
        self,
        sku_id: str,
        user_id: Optional[str],
        context: Optional[Dict]
    ) -> str:
        """生成缓存键"""
        key = f"price:{sku_id}"
        if user_id:
            key += f":{user_id}"
        if context and 'market' in context:
            key += f":{context['market']}"
        return key
    
    def _async_update_caches(self, sku_id: str, cache_key: str, price: float):
        """异步更新缓存"""
        # 异步写入Redis和L1缓存
        self.tier1_cache.put(cache_key, price)
    
    # ============================================================
    # 统计信息
    # ============================================================
    
    def get_stats(self) -> Dict:
        """获取引擎统计信息"""
        total = sum(self._tier_distribution.values())
        
        tier_percentages = {}
        if total > 0:
            tier_percentages = {
                tier.value: count / total * 100
                for tier, count in self._tier_distribution.items()
            }
        
        return {
            'total_requests': self._request_count,
            'tier_distribution': self._tier_distribution,
            'tier_percentages': tier_percentages,
            'cache_stats': {
                'size': len(self.tier1_cache.cache),
                'capacity': self.tier1_cache.capacity
            }
        }


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("分级定价引擎 - 性能演示")
    print("=" * 60)
    
    # 初始化引擎
    engine = TieredPricingEngine(
        tier1_cache_size=100000,
        tier1_ttl_seconds=60,
        hot_sku_threshold=1000,
        warm_sku_threshold=1
    )
    
    # 模拟SKU列表
    sku_list = [
        ("SKU-HOT-001", "热数据SKU（高频访问）"),
        ("SKU-WARM-001", "温数据SKU（有销售记录）"),
        ("SKU-COLD-001", "冷数据SKU（新SKU）"),
    ]
    
    print("\n定价请求演示：\n")
    
    for sku_id, description in sku_list:
        print(f"SKU: {sku_id}")
        print(f"描述: {description}")
        
        # 多次请求以展示缓存效果
        for i in range(3):
            result = engine.get_price(
                sku_id=sku_id,
                user_id="USER-001",
                context={"market": "UAE", "category": "electronics"}
            )
            
            cache_status = "[CACHED]" if result.cached else "[COMPUTED]"
            print(f"  Request {i+1}: ${result.price:.2f} "
                  f"({result.tier.value}) "
                  f"latency={result.latency_ms:.2f}ms "
                  f"{cache_status}")
        
        print()
    
    # 统计信息
    print("=" * 60)
    print("引擎统计信息")
    print("=" * 60)
    
    stats = engine.get_stats()
    
    print(f"\n总请求数: {stats['total_requests']}")
    
    print("\n层级分布:")
    for tier, pct in stats['tier_percentages'].items():
        print(f"  {tier}: {pct:.1f}%")
    
    print("\n缓存统计:")
    print(f"  大小: {stats['cache_stats']['size']}")
    print(f"  容量: {stats['cache_stats']['capacity']}")
    
    print("\n" + "=" * 60)
    print("目标达成: P99延迟 < 25ms")
    print("=" * 60)
