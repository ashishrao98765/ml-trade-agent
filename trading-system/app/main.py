from fastapi import FastAPI
import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List

import pandas as pd
import uvicorn

app = FastAPI(title="AI Trading System", version="1.0.0")

# In-memory storage (use database in production)
trading_data = []
performance_metrics = {"total_trades": 0, "profitable_trades": 0, "total_pnl": 0.0}

@app.get("/")
async def root():
    return {
        "service": "AI Trading System",
        "status": "operational",
        "total_trades": performance_metrics["total_trades"],
        "win_rate": calculate_win_rate(),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/webhook/tradingview")
async def tradingview_webhook(signal: dict):
    """Receive and process TradingView signals"""
    
    # Add timestamp if not present
    signal["received_at"] = datetime.now().isoformat()
    signal["processed"] = False
    
    # Store signal for learning
    trading_data.append(signal)
    
    # Process signal (simulate trade execution)
    result = await process_trade_signal(signal)
    
    # Log for monitoring
    print(f"📈 Signal: {signal['action']} {signal['symbol']} at {signal.get('price', 'market')}")
    
    return {
        "status": "received",
        "signal_id": len(trading_data),
        "action_taken": result["action"],
        "confidence": result.get("confidence", 0.75)
    }

async def process_trade_signal(signal: dict) -> dict:
    """Process trading signal with basic ML logic"""
    
    # Simple ML: Calculate confidence based on recent performance
    recent_trades = trading_data[-10:] if len(trading_data) >= 10 else trading_data
    
    if recent_trades:
        # Calculate success rate of similar signals
        similar_signals = [t for t in recent_trades if t.get("action") == signal["action"]]
        confidence = len(similar_signals) / len(recent_trades) if recent_trades else 0.5
    else:
        confidence = 0.5  # Default confidence for first trades
    
    # Simulate trade execution
    action_taken = "executed" if confidence > 0.3 else "skipped"
    
    # Update metrics
    performance_metrics["total_trades"] += 1
    if action_taken == "executed":
        # Simulate random profit/loss for demo
        import random
        pnl = random.uniform(-50, 100)  # Random P&L for demo
        performance_metrics["total_pnl"] += pnl
        if pnl > 0:
            performance_metrics["profitable_trades"] += 1
    
    signal["processed"] = True
    signal["action_taken"] = action_taken
    signal["confidence"] = confidence
    
    return {"action": action_taken, "confidence": confidence}

@app.get("/analytics")
async def get_analytics():
    """Get trading analytics and performance"""
    
    if not trading_data:
        return {"message": "No trading data available"}
    
    df = pd.DataFrame(trading_data)
    
    analytics = {
        "total_signals": len(trading_data),
        "buy_signals": len(df[df["action"] == "buy"]) if len(df) > 0 else 0,
        "sell_signals": len(df[df["action"] == "sell"]) if len(df) > 0 else 0,
        "win_rate": calculate_win_rate(),
        "total_pnl": performance_metrics["total_pnl"],
        "recent_signals": trading_data[-5:],  # Last 5 signals
        "performance": performance_metrics
    }
    
    return analytics

@app.get("/backtest")
async def run_backtest():
    """Simple backtesting on stored data"""
    
    if len(trading_data) < 10:
        return {"message": "Need at least 10 signals for backtesting"}
    
    # Simple backtest logic
    signals_df = pd.DataFrame(trading_data)
    
    # Calculate some basic metrics
    buy_signals = signals_df[signals_df["action"] == "buy"]
    sell_signals = signals_df[signals_df["action"] == "sell"]
    
    backtest_results = {
        "period": f"{len(trading_data)} signals analyzed",
        "total_buy_signals": len(buy_signals),
        "total_sell_signals": len(sell_signals),
        "avg_confidence": signals_df["confidence"].mean() if "confidence" in signals_df.columns else 0,
        "strategy_performance": "Learning from live data...",
        "recommendations": generate_recommendations()
    }
    
    return backtest_results

def calculate_win_rate() -> float:
    """Calculate win rate percentage"""
    if performance_metrics["total_trades"] == 0:
        return 0.0
    return (performance_metrics["profitable_trades"] / performance_metrics["total_trades"]) * 100

def generate_recommendations() -> List[str]:
    """Generate trading recommendations based on collected data"""
    recommendations = []
    
    if len(trading_data) > 0:
        df = pd.DataFrame(trading_data)
        
        # Analyze signal patterns
        if "rsi" in df.columns:
            avg_rsi = df["rsi"].mean()
            if avg_rsi > 70:
                recommendations.append("Market appears overbought - consider reducing buy signals")
            elif avg_rsi < 30:
                recommendations.append("Market appears oversold - consider increasing buy signals")
        
        # Analyze win rate
        win_rate = calculate_win_rate()
        if win_rate < 40:
            recommendations.append("Low win rate detected - consider adjusting strategy parameters")
        elif win_rate > 70:
            recommendations.append("High win rate - strategy performing well")
        
        recommendations.append(f"Collected {len(trading_data)} signals for ML training")
    
    return recommendations

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    