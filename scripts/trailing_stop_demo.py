"""
Trailing Stop-Loss Demo Script for Mizuto Trading Bot

This script demonstrates how to use the trailing stop-loss functionality
in both backtesting and live trading scenarios.
"""

import logging
import sys
import os

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.backtest import run_backtest

def demo_trailing_stop_backtest():
    """
    Demonstrates different trailing stop-loss configurations in backtesting.
    """
    print("=" * 60)
    print("TRAILING STOP-LOSS BACKTESTING DEMO")
    print("=" * 60)
    
    # Test parameters
    symbol = "BTC-USD"
    trade_amount = 1
    start_date = "2023-01-01"
    end_date = "2023-06-30"  # Shorter period for demo
    
    # Test different trailing stop configurations
    test_configs = [
        {
            "name": "No Stop-Loss",
            "trailing_stop_pct": None,
            "stop_loss_pct": None
        },
        {
            "name": "3% Trailing Stop",
            "trailing_stop_pct": 0.03,
            "stop_loss_pct": None
        },
        {
            "name": "5% Trailing Stop", 
            "trailing_stop_pct": 0.05,
            "stop_loss_pct": None
        },
        {
            "name": "10% Fixed Stop-Loss",
            "trailing_stop_pct": None,
            "stop_loss_pct": 0.10
        },
        {
            "name": "5% Trailing + 15% Fixed Stop",
            "trailing_stop_pct": 0.05,
            "stop_loss_pct": 0.15
        }
    ]
    
    for config in test_configs:
        print(f"\n{'='*20} {config['name']} {'='*20}")
        run_backtest(
            symbol=symbol,
            trade_amount=trade_amount,
            start_date=start_date,
            end_date=end_date,
            trailing_stop_pct=config['trailing_stop_pct'],
            stop_loss_pct=config['stop_loss_pct']
        )
        print("-" * 60)

def demo_live_trading_setup():
    """
    Shows how to set up trailing stop-loss for live trading.
    """
    print("\n" + "=" * 60)
    print("LIVE TRADING SETUP EXAMPLE")
    print("=" * 60)
    
    from src.bot import TradingBot
    
    # Example 1: Bot with 5% trailing stop-loss
    print("\nExample 1: Bot with 5% trailing stop-loss")
    bot1 = TradingBot(
        symbol="BTC-USD",
        trade_amount=0.01,
        trailing_stop_pct=0.05  # 5% trailing stop
    )
    print(f"Bot created with trailing stop: {bot1.trailing_stop_pct*100 if bot1.trailing_stop_pct else 'None'}%")
    
    # Example 2: Bot with both trailing and fixed stop-loss
    print("\nExample 2: Bot with 3% trailing stop + 10% fixed stop")
    bot2 = TradingBot(
        symbol="ETH-USD",
        trade_amount=0.1,
        trailing_stop_pct=0.03,  # 3% trailing stop
        stop_loss_pct=0.10       # 10% fixed stop
    )
    print(f"Bot created with:")
    print(f"  - Trailing stop: {bot2.trailing_stop_pct*100 if bot2.trailing_stop_pct else 'None'}%")
    print(f"  - Fixed stop: {bot2.stop_loss_pct*100 if bot2.stop_loss_pct else 'None'}%")
    
    # Example 3: Manual position tracking demonstration
    print("\nExample 3: Manual position tracking simulation")
    bot3 = TradingBot(
        symbol="AAPL",
        trade_amount=10,
        trailing_stop_pct=0.04  # 4% trailing stop
    )
    
    # Simulate entering a position
    entry_price = 150.00
    bot3.has_position = True
    bot3._handle_position_entry(entry_price)
    print(f"Position entered at: ${entry_price}")
    print(f"Initial stop-loss price: ${bot3.stop_loss_price:.2f}")
    
    # Simulate price movements
    price_movements = [152.00, 155.00, 158.00, 156.00, 153.00]
    
    for price in price_movements:
        signal = bot3._run_strategy_logic(price)
        print(f"Price: ${price:.2f} | Stop: ${bot3.stop_loss_price:.2f} | Signal: {signal}")
        
        if signal == 'sell':
            print(f"*** TRAILING STOP TRIGGERED AT ${price:.2f} ***")
            break

def explain_trailing_stop_concept():
    """
    Explains how trailing stop-loss works.
    """
    print("\n" + "=" * 60)
    print("HOW TRAILING STOP-LOSS WORKS")
    print("=" * 60)
    
    explanation = """
A trailing stop-loss is a dynamic stop-loss order that moves with the price
in your favor but stays fixed when the price moves against you.

Example with 5% trailing stop:
1. You buy at $100
2. Initial stop-loss is set at $95 (5% below entry)
3. Price rises to $110 → stop-loss moves to $104.50 (5% below $110)
4. Price rises to $120 → stop-loss moves to $114 (5% below $120)
5. Price falls to $115 → stop-loss stays at $114 (doesn't move down)
6. If price hits $114, position is sold

Benefits:
- Protects profits by locking in gains as price rises
- Limits losses if price moves against you
- Removes emotion from exit decisions
- Works automatically without constant monitoring

Configuration in Mizuto:
- trailing_stop_pct: Percentage below highest price (e.g., 0.05 = 5%)
- stop_loss_pct: Fixed percentage below entry price (optional)
- Can use both together (whichever triggers first)
"""
    print(explanation)

if __name__ == "__main__":
    # Configure logging for the demo
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("MIZUTO TRADING BOT - TRAILING STOP-LOSS DEMONSTRATION")
    print("This demo shows how to use trailing stop-loss in your trading bot.")
    
    # Explain the concept first
    explain_trailing_stop_concept()
    
    # Show live trading setup examples
    demo_live_trading_setup()
    
    # Ask user if they want to run backtests
    print("\n" + "=" * 60)
    response = input("Would you like to run backtest comparisons? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        demo_trailing_stop_backtest()
    else:
        print("Skipping backtest demonstrations.")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("You can now use trailing stop-loss in your trading bot by:")
    print("1. Setting trailing_stop_pct when creating TradingBot instance")
    print("2. Optionally setting stop_loss_pct for additional protection")
    print("3. Running backtests to optimize your stop-loss percentages")
    print("4. The bot will automatically handle position tracking and stop-loss logic")