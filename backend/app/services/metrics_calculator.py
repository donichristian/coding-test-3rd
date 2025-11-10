"""
Fund metrics calculator service
"""
from typing import Dict, Any, Optional
from decimal import Decimal
import numpy as np
import numpy_financial as npf
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.transaction import CapitalCall, Distribution, Adjustment


class MetricsCalculator:
    """Calculate fund performance metrics"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def calculate_all_metrics(self, fund_id: int) -> Dict[str, Any]:
        """Calculate all metrics for a fund"""
        pic = self.calculate_pic(fund_id)
        total_distributions = self.calculate_total_distributions(fund_id)
        dpi = self.calculate_dpi(fund_id)
        irr = self.calculate_irr(fund_id)
        
        return {
            "pic": float(pic) if pic else 0,
            "total_distributions": float(total_distributions) if total_distributions else 0,
            "dpi": float(dpi) if dpi else 0,
            "irr": float(irr) if irr else 0,
            "tvpi": None,  # To be implemented
            "rvpi": None,  # To be implemented
            "nav": None,   # To be implemented
        }
    
    def calculate_pic(self, fund_id: int) -> Optional[Decimal]:
        """
        Calculate Paid-In Capital (PIC)
        PIC = Total Capital Calls - Adjustments
        """
        # Get total capital calls
        total_calls = self.db.query(
            func.sum(CapitalCall.amount)
        ).filter(
            CapitalCall.fund_id == fund_id
        ).scalar() or Decimal(0)
        
        # Get total adjustments
        total_adjustments = self.db.query(
            func.sum(Adjustment.amount)
        ).filter(
            Adjustment.fund_id == fund_id
        ).scalar() or Decimal(0)
        
        pic = total_calls - total_adjustments
        return pic if pic > 0 else Decimal(0)
    
    def calculate_total_distributions(self, fund_id: int) -> Optional[Decimal]:
        """Calculate total distributions"""
        total = self.db.query(
            func.sum(Distribution.amount)
        ).filter(
            Distribution.fund_id == fund_id
        ).scalar() or Decimal(0)
        
        return total
    
    def calculate_dpi(self, fund_id: int) -> Optional[float]:
        """
        Calculate DPI (Distribution to Paid-In)
        DPI = Cumulative Distributions / PIC
        """
        pic = self.calculate_pic(fund_id)
        total_distributions = self.calculate_total_distributions(fund_id)

        print(f"DEBUG DPI: PIC = {pic}, Total Distributions = {total_distributions}")

        if not pic or pic == 0:
            print("DEBUG DPI: PIC is 0 or None, returning 0.0")
            return 0.0

        if not total_distributions or total_distributions == 0:
            print("DEBUG DPI: Total distributions is 0 or None, returning 0.0")
            return 0.0

        dpi = float(total_distributions) / float(pic)
        result = round(dpi, 4)
        print(f"DEBUG DPI: {total_distributions} / {pic} = {result}")
        return result
    
    def calculate_irr(self, fund_id: int) -> Optional[float]:
        """
        Calculate IRR (Internal Rate of Return)
        Uses numpy-financial's irr function
        """
        try:
            # Get all cash flows sorted by date
            cash_flows = self._get_cash_flows(fund_id)
            
            if len(cash_flows) < 2:
                return None
            
            # Extract amounts
            amounts = [cf['amount'] for cf in cash_flows]
            
            # Calculate IRR (returns as decimal, e.g., 0.15 for 15%)
            irr = npf.irr(amounts)
            
            if irr is None or np.isnan(irr) or np.isinf(irr):
                return None
            
            # Convert to percentage
            return round(float(irr) * 100, 2)
            
        except Exception as e:
            print(f"Error calculating IRR: {e}")
            return None
    
    def _get_cash_flows(self, fund_id: int) -> list:
        """
        Get all cash flows for IRR calculation
        Capital calls are negative, distributions are positive
        """
        cash_flows = []
        
        # Get capital calls (negative cash flows)
        calls = self.db.query(
            CapitalCall.call_date,
            CapitalCall.amount
        ).filter(
            CapitalCall.fund_id == fund_id
        ).order_by(
            CapitalCall.call_date
        ).all()
        
        for call in calls:
            cash_flows.append({
                'date': call.call_date,
                'amount': -float(call.amount),  # Negative for outflow
                'type': 'capital_call'
            })
        
        # Get distributions (positive cash flows)
        distributions = self.db.query(
            Distribution.distribution_date,
            Distribution.amount
        ).filter(
            Distribution.fund_id == fund_id
        ).order_by(
            Distribution.distribution_date
        ).all()
        
        for dist in distributions:
            cash_flows.append({
                'date': dist.distribution_date,
                'amount': float(dist.amount),  # Positive for inflow
                'type': 'distribution'
            })
        
        # Sort by date
        cash_flows.sort(key=lambda x: x['date'])
        
        return cash_flows
    
    def get_calculation_breakdown(self, fund_id: int, metric: str) -> Dict[str, Any]:
        """
        Get detailed breakdown of a calculation with cash flows for debugging
        
        Args:
            fund_id: Fund ID
            metric: Metric name (dpi, irr, pic)
            
        Returns:
            Detailed breakdown with intermediate values and transaction details
        """
        if metric == "dpi":
            pic = self.calculate_pic(fund_id)
            total_distributions = self.calculate_total_distributions(fund_id)
            dpi = self.calculate_dpi(fund_id)
            
            # Get detailed transactions for debugging
            capital_calls = self.db.query(CapitalCall).filter(
                CapitalCall.fund_id == fund_id
            ).order_by(CapitalCall.call_date).all()
            
            distributions = self.db.query(Distribution).filter(
                Distribution.fund_id == fund_id
            ).order_by(Distribution.distribution_date).all()
            
            adjustments = self.db.query(Adjustment).filter(
                Adjustment.fund_id == fund_id
            ).order_by(Adjustment.adjustment_date).all()
            
            return {
                "metric": "DPI",
                "formula": "Cumulative Distributions / Paid-In Capital",
                "pic": float(pic) if pic else 0,
                "total_distributions": float(total_distributions) if total_distributions else 0,
                "result": dpi,
                "explanation": f"DPI = {total_distributions} / {pic} = {dpi}",
                "transactions": {
                    "capital_calls": [
                        {
                            "date": str(call.call_date),
                            "amount": float(call.amount),
                            "description": call.description
                        } for call in capital_calls
                    ],
                    "distributions": [
                        {
                            "date": str(dist.distribution_date),
                            "amount": float(dist.amount),
                            "is_recallable": dist.is_recallable,
                            "description": dist.description
                        } for dist in distributions
                    ],
                    "adjustments": [
                        {
                            "date": str(adj.adjustment_date),
                            "amount": float(adj.amount),
                            "type": adj.adjustment_type,
                            "description": adj.description
                        } for adj in adjustments
                    ]
                }
            }
        
        elif metric == "irr":
            cash_flows = self._get_cash_flows(fund_id)
            irr = self.calculate_irr(fund_id)
            
            return {
                "metric": "IRR",
                "formula": "Internal Rate of Return (NPV = 0)",
                "cash_flows": cash_flows,
                "result": irr,
                "explanation": f"IRR calculated from {len(cash_flows)} cash flows = {irr}%",
                "cash_flow_summary": {
                    "total_outflows": sum(cf['amount'] for cf in cash_flows if cf['amount'] < 0),
                    "total_inflows": sum(cf['amount'] for cf in cash_flows if cf['amount'] > 0),
                    "net_cash_flow": sum(cf['amount'] for cf in cash_flows)
                }
            }
        
        elif metric == "pic":
            # Get detailed capital calls
            capital_calls = self.db.query(CapitalCall).filter(
                CapitalCall.fund_id == fund_id
            ).order_by(CapitalCall.call_date).all()
            
            # Get detailed adjustments
            adjustments = self.db.query(Adjustment).filter(
                Adjustment.fund_id == fund_id
            ).order_by(Adjustment.adjustment_date).all()
            
            total_calls = sum(float(call.amount) for call in capital_calls)
            total_adjustments = sum(float(adj.amount) for adj in adjustments)
            pic = self.calculate_pic(fund_id)
            
            return {
                "metric": "PIC",
                "formula": "Total Capital Calls - Adjustments",
                "total_calls": total_calls,
                "total_adjustments": total_adjustments,
                "result": float(pic) if pic else 0,
                "explanation": f"PIC = {total_calls} - {total_adjustments} = {pic}",
                "transactions": {
                    "capital_calls": [
                        {
                            "date": str(call.call_date),
                            "amount": float(call.amount),
                            "description": call.description
                        } for call in capital_calls
                    ],
                    "adjustments": [
                        {
                            "date": str(adj.adjustment_date),
                            "amount": float(adj.amount),
                            "type": adj.adjustment_type,
                            "description": adj.description
                        } for adj in adjustments
                    ]
                }
            }
        
        return {"error": "Unknown metric"}
