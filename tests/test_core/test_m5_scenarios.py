"""M5 情景测试 — 真实场景下的路径权重变化。

每个测试对应一个可辨识的地缘政治情景，
验证系统反应的方向和量级是否合理。
"""
import pytest
from hormuz.core.types import PathWeights
from hormuz.core.m5_game import adjust_path_weights, SignalEvidence


# 三种典型 base weight（来自 MC 物理模拟）
BASE_NEUTRAL = PathWeights(a=0.30, b=0.50, c=0.20)   # 中性
BASE_HAWKISH = PathWeights(a=0.15, b=0.45, c=0.40)   # 偏鹰（H2 主导）
BASE_DOVISH  = PathWeights(a=0.50, b=0.35, c=0.15)   # 偏鸽（H1 主导）


def _pct(pw: PathWeights) -> str:
    return f"A={pw.a:.0%} B={pw.b:.0%} C={pw.c:.0%}"


def _shift(base: PathWeights, result: PathWeights) -> str:
    da = (result.a - base.a) * 100
    db = (result.b - base.b) * 100
    dc = (result.c - base.c) * 100
    return f"ΔA={da:+.1f}pp ΔB={db:+.1f}pp ΔC={dc:+.1f}pp"


# ═══════════════════════════════════════════════════════════════════════
# 情景 1: 平静日 — 无信号
# ═══════════════════════════════════════════════════════════════════════

class TestQuietDay:
    """无信号触发的普通日 → 路径不变"""

    def test_no_change(self):
        result = adjust_path_weights(BASE_NEUTRAL, [])
        assert result.a == pytest.approx(BASE_NEUTRAL.a)
        assert result.b == pytest.approx(BASE_NEUTRAL.b)


# ═══════════════════════════════════════════════════════════════════════
# 情景 2: 阿曼秘密传话 — cheap talk, low evidence
# ═══════════════════════════════════════════════════════════════════════

class TestOmanBackchannel:
    """阿曼通过秘密渠道传话，新闻仅有模糊提及。
    预期：几乎不动（<2pp），因为 cheap talk + low evidence。"""

    def test_tiny_shift(self):
        signals = [SignalEvidence("external_mediation", 0.2)]
        result = adjust_path_weights(BASE_NEUTRAL, signals)
        assert abs(result.a - BASE_NEUTRAL.a) < 0.02

    def test_c_barely_drops(self):
        signals = [SignalEvidence("external_mediation", 0.2)]
        result = adjust_path_weights(BASE_NEUTRAL, signals)
        assert abs(result.c - BASE_NEUTRAL.c) < 0.02


# ═══════════════════════════════════════════════════════════════════════
# 情景 3: 中国公开担保 + 冻结资产 — costly, high evidence
# ═══════════════════════════════════════════════════════════════════════

class TestChinaGuarantee:
    """中国公开声明担保，冻结伊朗在华资产作为承诺。
    这是 costly_self_binding + high evidence。"""

    def test_significant_a_boost(self):
        signals = [SignalEvidence("costly_self_binding", 1.0)]
        result = adjust_path_weights(BASE_NEUTRAL, signals)
        shift = result.a - BASE_NEUTRAL.a
        assert 0.03 < shift < 0.15

    def test_c_drops(self):
        signals = [SignalEvidence("costly_self_binding", 1.0)]
        result = adjust_path_weights(BASE_NEUTRAL, signals)
        assert result.c < BASE_NEUTRAL.c

    def test_works_on_hawkish_base(self):
        """即使 base 偏鹰，costly commitment 仍能推 A。"""
        signals = [SignalEvidence("costly_self_binding", 1.0)]
        result = adjust_path_weights(BASE_HAWKISH, signals)
        assert result.a > BASE_HAWKISH.a


# ═══════════════════════════════════════════════════════════════════════
# 情景 4: IRGC 攻击油港 — costly escalation, high evidence
# ═══════════════════════════════════════════════════════════════════════

class TestIRGCOilTerminalAttack:
    """IRGC 导弹攻击 Ras Tanura 油港，卫星照片确认。
    不可逆，高可见度。"""

    def test_c_surges(self):
        signals = [SignalEvidence("irgc_escalation", 1.0)]
        result = adjust_path_weights(BASE_NEUTRAL, signals)
        shift_c = result.c - BASE_NEUTRAL.c
        assert shift_c > 0.02  # C 涨 2+pp（单条信号不应太猛）

    def test_a_drops_hard(self):
        signals = [SignalEvidence("irgc_escalation", 1.0)]
        result = adjust_path_weights(BASE_NEUTRAL, signals)
        assert result.a < BASE_NEUTRAL.a

    def test_low_evidence_modest(self):
        """仅有未确认的社交媒体传言 → 小幅变化"""
        signals = [SignalEvidence("irgc_escalation", 0.2)]
        result = adjust_path_weights(BASE_NEUTRAL, signals)
        shift_c = result.c - BASE_NEUTRAL.c
        assert shift_c < 0.04


# ═══════════════════════════════════════════════════════════════════════
# 情景 5: 焦点收敛 — 斡旋 + 承诺同时出现
# ═══════════════════════════════════════════════════════════════════════

class TestFocalConvergence:
    """卡塔尔斡旋 + 伊朗释放扣押油轮。
    两条 A 方向信号同时触发 → 焦点收敛（涌现，不需要额外信号）。"""

    def test_convergence_stronger_than_sum(self):
        """非线性放大：合力 > 各项之和"""
        sig_med = [SignalEvidence("external_mediation", 1.0)]
        sig_bind = [SignalEvidence("costly_self_binding", 1.0)]
        sig_all = [
            SignalEvidence("external_mediation", 1.0),
            SignalEvidence("costly_self_binding", 1.0),
        ]
        r_med = adjust_path_weights(BASE_NEUTRAL, sig_med)
        r_bind = adjust_path_weights(BASE_NEUTRAL, sig_bind)
        r_all = adjust_path_weights(BASE_NEUTRAL, sig_all)

        sum_shifts = (r_med.a - BASE_NEUTRAL.a) + (r_bind.a - BASE_NEUTRAL.a)
        combined = r_all.a - BASE_NEUTRAL.a
        assert combined > sum_shifts

    def test_a_rises_significantly(self):
        """两条同向信号 → A 应该明显上升"""
        signals = [
            SignalEvidence("external_mediation", 1.0),
            SignalEvidence("costly_self_binding", 1.0),
        ]
        result = adjust_path_weights(BASE_NEUTRAL, signals)
        assert result.a > BASE_NEUTRAL.a + 0.04


# ═══════════════════════════════════════════════════════════════════════
# 情景 6: 矛盾信号 — 斡旋 + 升级同时出现
# ═══════════════════════════════════════════════════════════════════════

class TestContradictorySignals:
    """卡塔尔在斡旋的同时，IRGC 攻击了油轮。
    两个方向相反的信号应部分抵消。"""

    def test_partial_cancel(self):
        signals = [
            SignalEvidence("external_mediation", 0.8),
            SignalEvidence("irgc_escalation", 0.8),
        ]
        result = adjust_path_weights(BASE_NEUTRAL, signals)

        r_med_only = adjust_path_weights(
            BASE_NEUTRAL, [SignalEvidence("external_mediation", 0.8)])
        r_esc_only = adjust_path_weights(
            BASE_NEUTRAL, [SignalEvidence("irgc_escalation", 0.8)])

        # A 不如纯斡旋高，C 不如纯升级高
        assert result.a < r_med_only.a
        assert result.c < r_esc_only.c

    def test_b_absorbs_uncertainty(self):
        """矛盾信号 → B 应该相对增大（不确定性增加）"""
        signals = [
            SignalEvidence("external_mediation", 0.8),
            SignalEvidence("irgc_escalation", 0.8),
        ]
        result = adjust_path_weights(BASE_NEUTRAL, signals)
        # B should not drop much — uncertainty absorber
        assert result.b >= BASE_NEUTRAL.b * 0.9


# ═══════════════════════════════════════════════════════════════════════
# 情景 7: 全面升级 — 多个升级信号
# ═══════════════════════════════════════════════════════════════════════

class TestFullEscalation:
    """IRGC 攻击油港 + 美国内部政策分裂 → C 路径主导"""

    def test_c_becomes_dominant(self):
        signals = [
            SignalEvidence("irgc_escalation", 1.0),
            SignalEvidence("us_inconsistency", 0.8),
        ]
        result = adjust_path_weights(BASE_NEUTRAL, signals)
        assert result.c > BASE_NEUTRAL.c
        assert result.a < BASE_NEUTRAL.a

    def test_on_hawkish_base_c_rises(self):
        """偏鹰 base + 升级信号 → C 应该上升"""
        signals = [
            SignalEvidence("irgc_escalation", 1.0),
            SignalEvidence("us_inconsistency", 0.8),
        ]
        result = adjust_path_weights(BASE_HAWKISH, signals)
        assert result.c > BASE_HAWKISH.c


# ═══════════════════════════════════════════════════════════════════════
# 情景 8: 边际信号 — medium evidence 日常噪音
# ═══════════════════════════════════════════════════════════════════════

class TestRoutineNoise:
    """普通日的 medium evidence 斡旋 + 低证据升级 → 小幅变化"""

    def test_total_shift_small(self):
        signals = [
            SignalEvidence("external_mediation", 0.5),
            SignalEvidence("irgc_escalation", 0.2),
        ]
        result = adjust_path_weights(BASE_NEUTRAL, signals)
        total_shift = (
            abs(result.a - BASE_NEUTRAL.a)
            + abs(result.b - BASE_NEUTRAL.b)
            + abs(result.c - BASE_NEUTRAL.c)
        )
        # Total absolute shift < 10pp (路径权重变化的总绝对值)
        assert total_shift < 0.10


# ═══════════════════════════════════════════════════════════════════════
# 情景 9: IRGC 内部分裂 — combo 信号
# ═══════════════════════════════════════════════════════════════════════

class TestIRGCFragmentation:
    """美国发出矛盾信号（prerequisite）+ IRGC 内部权力斗争曝光。"""

    def test_fragmentation_needs_us_inconsistency(self):
        """没有 us_inconsistency 前置，fragmentation 不触发"""
        signals = [SignalEvidence("irgc_fragmentation", 1.0)]
        result = adjust_path_weights(BASE_NEUTRAL, signals)
        assert result.a == pytest.approx(BASE_NEUTRAL.a)

    def test_combo_fires(self):
        signals = [
            SignalEvidence("us_inconsistency", 0.8),
            SignalEvidence("irgc_fragmentation", 1.0),
        ]
        result = adjust_path_weights(BASE_NEUTRAL, signals)
        assert result.a > BASE_NEUTRAL.a


# ═══════════════════════════════════════════════════════════════════════
# 打印报告（pytest -s 时可见）
# ═══════════════════════════════════════════════════════════════════════

def test_scenario_report(capsys):
    """打印所有情景的实际数值（pytest -s 查看）"""
    scenarios = [
        ("平静日", []),
        ("阿曼秘密传话 (low)", [("external_mediation", 0.2)]),
        ("阿曼公开斡旋 (high)", [("external_mediation", 1.0)]),
        ("中国公开担保 (high)", [("costly_self_binding", 1.0)]),
        ("IRGC攻击油港 (high)", [("irgc_escalation", 1.0)]),
        ("IRGC传言 (low)", [("irgc_escalation", 0.2)]),
        ("焦点收敛: 斡旋+承诺 (2×high)", [
            ("external_mediation", 1.0),
            ("costly_self_binding", 1.0),
        ]),
        ("矛盾: 斡旋+升级", [
            ("external_mediation", 0.8),
            ("irgc_escalation", 0.8),
        ]),
        ("全面升级", [
            ("irgc_escalation", 1.0),
            ("us_inconsistency", 0.8),
        ]),
        ("日常噪音", [
            ("external_mediation", 0.5),
            ("irgc_escalation", 0.2),
        ]),
    ]

    print("\n" + "=" * 72)
    print("M5 情景测试报告")
    print(f"Base: {_pct(BASE_NEUTRAL)}")
    print("=" * 72)

    for name, sigs in scenarios:
        signals = [SignalEvidence(k, e) for k, e in sigs]
        result = adjust_path_weights(BASE_NEUTRAL, signals)
        print(f"\n{name}")
        print(f"  结果: {_pct(result)}")
        print(f"  变化: {_shift(BASE_NEUTRAL, result)}")

    print("\n" + "-" * 72)
    print("偏鹰 base 测试 (A=15% B=45% C=40%)")
    print("-" * 72)
    for name, sigs in [
        ("中国公开担保", [("costly_self_binding", 1.0)]),
        ("IRGC攻击油港", [("irgc_escalation", 1.0)]),
        ("焦点收敛: 斡旋+承诺", [
            ("external_mediation", 1.0),
            ("costly_self_binding", 1.0),
        ]),
    ]:
        signals = [SignalEvidence(k, e) for k, e in sigs]
        result = adjust_path_weights(BASE_HAWKISH, signals)
        print(f"\n{name}")
        print(f"  结果: {_pct(result)}")
        print(f"  变化: {_shift(BASE_HAWKISH, result)}")

    print("\n" + "=" * 72)
