"""
Unit Tests for Alpha Model
GPT 5.2 Codex Audit Implementation

Tests:
1. Market Impact Model (Almgren-Chriss)
2. Survivorship Bias Handler
3. Integrated Alpha Model
"""

import pytest
import sys
import os
from datetime import date, datetime
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from shield.alpha_model import (
    MarketImpactModel,
    MarketImpactParams,
    ImpactEstimate,
    SurvivorshipBiasHandler,
    DelistingReason,
    DelistedStock,
    SurvivorshipBiasReport,
    AlphaModel,
    AlphaSignal
)


class TestMarketImpactModel:
    """Tests for Market Impact Model"""
    
    def test_initialization(self):
        """Test model initialization with default params"""
        model = MarketImpactModel()
        assert model.params.gamma == 0.1
        assert model.params.eta == 0.01
        assert model.params.max_participation_rate == 0.10
    
    def test_custom_params(self):
        """Test model initialization with custom params"""
        params = MarketImpactParams(
            gamma=0.2,
            eta=0.02,
            sigma=0.30
        )
        model = MarketImpactModel(params)
        assert model.params.gamma == 0.2
        assert model.params.eta == 0.02
        assert model.params.sigma == 0.30
    
    def test_set_stock_data(self):
        """Test setting stock-specific data"""
        model = MarketImpactModel()
        model.set_stock_data("7203", adv=5_000_000, volatility=0.25)
        
        assert "7203" in model._adv_cache
        assert model._adv_cache["7203"] == 5_000_000
        assert model._volatility_cache["7203"] == 0.25
    
    def test_permanent_impact_calculation(self):
        """Test permanent impact calculation"""
        model = MarketImpactModel()
        
        # 10% participation should give meaningful impact
        impact = model.calculate_permanent_impact(
            order_size=500_000,
            adv=5_000_000,
            price=2500
        )
        
        assert impact > 0
        assert impact < 1000  # Should be reasonable (< 10%)
    
    def test_temporary_impact_calculation(self):
        """Test temporary impact calculation"""
        model = MarketImpactModel()
        
        impact = model.calculate_temporary_impact(
            order_size=500_000,
            adv=5_000_000,
            price=2500,
            execution_horizon_days=1.0
        )
        
        assert impact > 0
        assert impact < 1000
    
    def test_spread_cost(self):
        """Test spread cost calculation"""
        model = MarketImpactModel()
        spread = model.calculate_spread_cost(2500)
        
        # Default spread is 10bps, half spread is 5bps
        assert spread == 5.0
    
    def test_total_impact_estimate(self):
        """Test total impact estimation"""
        model = MarketImpactModel()
        model.set_stock_data("7203", adv=5_000_000, volatility=0.25)
        
        impact = model.estimate_total_impact(
            code="7203",
            order_size=100_000,
            price=2500,
            side="BUY",
            urgency="NORMAL"
        )
        
        assert isinstance(impact, ImpactEstimate)
        assert impact.permanent_impact_bps >= 0
        assert impact.temporary_impact_bps >= 0
        assert impact.total_impact_bps > 0
        assert impact.execution_cost_jpy > 0
        assert impact.participation_rate == 100_000 / 5_000_000
        assert impact.confidence == 0.9  # Data available
    
    def test_impact_without_adv_data(self):
        """Test impact estimation without ADV data"""
        model = MarketImpactModel()
        
        impact = model.estimate_total_impact(
            code="9999",  # Unknown stock
            order_size=10_000,
            price=1000,
            side="BUY",
            urgency="NORMAL"
        )
        
        assert impact.confidence == 0.6  # Lower confidence
        assert len(impact.warnings) > 0  # Should have warning
    
    def test_participation_rate_warning(self):
        """Test warning for high participation rate"""
        model = MarketImpactModel()
        model.set_stock_data("7203", adv=100_000, volatility=0.25)
        
        # Order larger than 10% of ADV
        impact = model.estimate_total_impact(
            code="7203",
            order_size=20_000,  # 20% of ADV
            price=2500,
            side="BUY",
            urgency="NORMAL"
        )
        
        assert any("exceeds limit" in w for w in impact.warnings)
    
    def test_execution_schedule(self):
        """Test optimal execution schedule calculation"""
        model = MarketImpactModel()
        model.set_stock_data("7203", adv=1_000_000, volatility=0.25)
        
        schedule = model.calculate_optimal_execution_schedule(
            code="7203",
            total_shares=500_000,
            price=2500,
            max_days=5
        )
        
        assert len(schedule) > 0
        assert all("day" in s for s in schedule)
        assert all("shares" in s for s in schedule)
        
        # Total shares should equal input
        total_scheduled = sum(s["shares"] for s in schedule)
        assert total_scheduled == 500_000


class TestSurvivorshipBiasHandler:
    """Tests for Survivorship Bias Handler"""
    
    def test_initialization(self):
        """Test handler initialization"""
        handler = SurvivorshipBiasHandler()
        assert len(handler._delisted_stocks) == 0
        assert len(handler._universe_history) == 0
    
    def test_register_bankruptcy(self):
        """Test registering a bankruptcy"""
        handler = SurvivorshipBiasHandler()
        
        delisted = handler.register_delisting(
            code="9999",
            name="Failed Corp",
            delisting_date=date(2020, 3, 15),
            reason=DelistingReason.BANKRUPTCY,
            final_price=10,
            last_held_price=1000
        )
        
        assert delisted.code == "9999"
        assert delisted.reason == DelistingReason.BANKRUPTCY
        assert delisted.final_return == -1.0  # Total loss
    
    def test_register_merger(self):
        """Test registering a merger"""
        handler = SurvivorshipBiasHandler()
        
        delisted = handler.register_delisting(
            code="8888",
            name="Merged Inc",
            delisting_date=date(2021, 6, 30),
            reason=DelistingReason.MERGER,
            final_price=1500,
            last_held_price=1200,
            merger_ratio=1.2,
            acquirer_code="7777"
        )
        
        assert delisted.reason == DelistingReason.MERGER
        assert delisted.merger_ratio == 1.2
        # Return should be positive due to merger premium
        expected_return = (1500 * 1.2 - 1200) / 1200
        assert abs(delisted.final_return - expected_return) < 0.01
    
    def test_is_delisted(self):
        """Test delisting check"""
        handler = SurvivorshipBiasHandler()
        
        handler.register_delisting(
            code="9999",
            name="Failed Corp",
            delisting_date=date(2020, 3, 15),
            reason=DelistingReason.BANKRUPTCY,
            final_price=10
        )
        
        # Before delisting
        assert not handler.is_delisted("9999", date(2020, 3, 1))
        
        # On delisting date
        assert handler.is_delisted("9999", date(2020, 3, 15))
        
        # After delisting
        assert handler.is_delisted("9999", date(2020, 4, 1))
        
        # Unknown stock
        assert not handler.is_delisted("1111", date(2020, 4, 1))
    
    def test_universe_recording(self):
        """Test universe snapshot recording"""
        handler = SurvivorshipBiasHandler()
        
        handler.record_universe(date(2020, 1, 1), ["7203", "9984", "6758"])
        handler.record_universe(date(2020, 6, 1), ["7203", "9984"])  # 6758 removed
        
        # Get universe at different dates
        universe_jan = handler.get_universe(date(2020, 3, 1))
        assert "6758" in universe_jan
        
        universe_jul = handler.get_universe(date(2020, 7, 1))
        assert "6758" not in universe_jul
    
    def test_position_delisting_bankruptcy(self):
        """Test handling position in bankrupt company"""
        handler = SurvivorshipBiasHandler()
        
        handler.register_delisting(
            code="9999",
            name="Failed Corp",
            delisting_date=date(2020, 3, 15),
            reason=DelistingReason.BANKRUPTCY,
            final_price=10
        )
        
        adjustment = handler.handle_position_delisting(
            code="9999",
            position_size=1000,
            entry_price=800,
            delisting_date=date(2020, 3, 15)
        )
        
        assert adjustment["adjustment_type"] == "TERMINAL_VALUE"
        assert adjustment["terminal_multiplier"] == 0.0  # Bankruptcy = total loss
        assert adjustment["exit_value"] == 0
        assert adjustment["pnl"] == -800_000  # Total loss
    
    def test_position_delisting_acquisition(self):
        """Test handling position in acquired company"""
        handler = SurvivorshipBiasHandler()
        
        handler.register_delisting(
            code="8888",
            name="Acquired Inc",
            delisting_date=date(2021, 6, 30),
            reason=DelistingReason.ACQUISITION,
            final_price=1500
        )
        
        adjustment = handler.handle_position_delisting(
            code="8888",
            position_size=1000,
            entry_price=1200,
            delisting_date=date(2021, 6, 30)
        )
        
        assert adjustment["adjustment_type"] == "ACQUISITION_CASH"
        assert adjustment["exit_value"] == 1500 * 1000
        assert adjustment["pnl"] > 0  # Profit from acquisition
    
    def test_bias_adjustment_calculation(self):
        """Test survivorship bias adjustment calculation"""
        handler = SurvivorshipBiasHandler()
        
        # Register some delistings
        handler.register_delisting(
            code="9999",
            name="Failed Corp",
            delisting_date=date(2020, 3, 15),
            reason=DelistingReason.BANKRUPTCY,
            final_price=10
        )
        
        # Handle position
        handler.handle_position_delisting(
            code="9999",
            position_size=1000,
            entry_price=800,
            delisting_date=date(2020, 3, 15)
        )
        
        # Calculate bias adjustment
        report = handler.calculate_bias_adjustment(
            backtest_start=date(2019, 1, 1),
            backtest_end=date(2022, 12, 31),
            total_return=0.50
        )
        
        assert isinstance(report, SurvivorshipBiasReport)
        assert report.delisted_count == 1
        assert report.bankruptcy_count == 1
        assert report.bias_adjustment_factor > 1.0  # Should show bias
    
    def test_audit_trail(self):
        """Test audit trail generation"""
        handler = SurvivorshipBiasHandler()
        
        handler.register_delisting(
            code="9999",
            name="Failed Corp",
            delisting_date=date(2020, 3, 15),
            reason=DelistingReason.BANKRUPTCY,
            final_price=10
        )
        
        audit = handler.get_audit_trail()
        
        assert "delisted_stocks" in audit
        assert "9999" in audit["delisted_stocks"]
        assert "methodology" in audit
        assert "terminal_values" in audit["methodology"]


class TestAlphaModel:
    """Tests for integrated Alpha Model"""
    
    def test_initialization(self):
        """Test Alpha Model initialization"""
        model = AlphaModel()
        
        assert model.max_impact_bps == 50.0
        assert model.min_alpha_threshold == 0.01
        assert isinstance(model.impact_model, MarketImpactModel)
        assert isinstance(model.bias_handler, SurvivorshipBiasHandler)
    
    def test_set_market_data(self):
        """Test setting market data"""
        model = AlphaModel()
        model.set_market_data("7203", adv=5_000_000, volatility=0.25)
        
        assert "7203" in model.impact_model._adv_cache
    
    def test_generate_signal_basic(self):
        """Test basic signal generation"""
        model = AlphaModel()
        model.set_market_data("7203", adv=5_000_000, volatility=0.25)
        
        signal = model.generate_signal(
            code="7203",
            raw_alpha=0.15,
            price=2500,
            target_value=50_000_000,
            as_of_date=date(2026, 1, 29),
            urgency="NORMAL"
        )
        
        assert signal is not None
        assert isinstance(signal, AlphaSignal)
        assert signal.raw_alpha == 0.15
        assert signal.adjusted_alpha < signal.raw_alpha  # Impact reduces alpha
        assert signal.recommended_size > 0
    
    def test_signal_rejected_low_alpha(self):
        """Test signal rejection for low alpha"""
        model = AlphaModel(min_alpha_threshold=0.05)
        model.set_market_data("7203", adv=5_000_000, volatility=0.25)
        
        signal = model.generate_signal(
            code="7203",
            raw_alpha=0.02,  # Below threshold
            price=2500,
            target_value=50_000_000,
            as_of_date=date(2026, 1, 29)
        )
        
        assert signal is None
    
    def test_signal_rejected_delisted(self):
        """Test signal rejection for delisted stock"""
        model = AlphaModel()
        
        # Register delisting
        model.bias_handler.register_delisting(
            code="9999",
            name="Failed Corp",
            delisting_date=date(2020, 3, 15),
            reason=DelistingReason.BANKRUPTCY,
            final_price=10
        )
        
        # Try to generate signal after delisting
        signal = model.generate_signal(
            code="9999",
            raw_alpha=0.20,
            price=10,
            target_value=10_000_000,
            as_of_date=date(2020, 4, 1)  # After delisting
        )
        
        assert signal is None
    
    def test_signal_size_reduction_for_impact(self):
        """Test that signal size is reduced when impact exceeds limit"""
        model = AlphaModel(max_impact_bps=10.0)  # Low limit
        model.set_market_data("7203", adv=100_000, volatility=0.30)  # Low liquidity
        
        signal = model.generate_signal(
            code="7203",
            raw_alpha=0.20,
            price=2500,
            target_value=100_000_000,  # Large order
            as_of_date=date(2026, 1, 29)
        )
        
        if signal:
            # Size should be reduced to meet impact limit
            assert signal.impact_cost_bps <= 10.0
    
    def test_execution_schedule(self):
        """Test execution schedule generation"""
        model = AlphaModel()
        model.set_market_data("7203", adv=1_000_000, volatility=0.25)
        
        schedule = model.get_execution_schedule(
            code="7203",
            total_shares=500_000,
            price=2500
        )
        
        assert len(schedule) > 0
    
    def test_signal_history(self):
        """Test signal history tracking"""
        model = AlphaModel()
        model.set_market_data("7203", adv=5_000_000, volatility=0.25)
        
        # Generate some signals
        for i in range(3):
            model.generate_signal(
                code="7203",
                raw_alpha=0.10 + i * 0.05,
                price=2500,
                target_value=10_000_000,
                as_of_date=date(2026, 1, 29)
            )
        
        history = model.get_signal_history()
        assert len(history) == 3
    
    def test_audit_report(self):
        """Test audit report generation"""
        model = AlphaModel()
        model.set_market_data("7203", adv=5_000_000, volatility=0.25)
        
        # Generate a signal
        model.generate_signal(
            code="7203",
            raw_alpha=0.15,
            price=2500,
            target_value=50_000_000,
            as_of_date=date(2026, 1, 29)
        )
        
        audit = model.get_audit_report()
        
        assert "model_version" in audit
        assert "impact_model" in audit
        assert "survivorship_bias" in audit
        assert audit["signal_count"] == 1


class TestIntegration:
    """Integration tests"""
    
    def test_full_workflow(self):
        """Test complete workflow from signal to execution"""
        # Initialize model
        model = AlphaModel(
            max_impact_bps=50.0,
            min_alpha_threshold=0.02
        )
        
        # Set up market data
        codes = ["7203", "9984", "6758"]
        for code in codes:
            model.set_market_data(code, adv=3_000_000, volatility=0.25)
        
        # Register a delisting
        model.bias_handler.register_delisting(
            code="9999",
            name="Failed Corp",
            delisting_date=date(2020, 3, 15),
            reason=DelistingReason.BANKRUPTCY,
            final_price=10
        )
        
        # Generate signals
        signals = []
        for code in codes:
            signal = model.generate_signal(
                code=code,
                raw_alpha=0.12,
                price=2500,
                target_value=30_000_000,
                as_of_date=date(2026, 1, 29)
            )
            if signal:
                signals.append(signal)
        
        assert len(signals) == 3
        
        # Get execution schedules
        for signal in signals:
            schedule = model.get_execution_schedule(
                code=signal.code,
                total_shares=signal.recommended_size,
                price=2500
            )
            assert len(schedule) > 0
        
        # Get bias report
        report = model.get_bias_report(
            start_date=date(2019, 1, 1),
            end_date=date(2026, 12, 31),
            total_return=0.50
        )
        assert report.delisted_count == 1
        
        # Get audit report
        audit = model.get_audit_report()
        assert audit["signal_count"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
