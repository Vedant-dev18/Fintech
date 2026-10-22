
from decimal import Decimal


def calculate_savings(expenses, reduction_percent):
    expenses = Decimal(str(expenses))
    reduction = Decimal(str(reduction_percent))

    if expenses < 0:
        raise ValueError("Expenses cannot be negative")

    if not 0 <= reduction <= 100:
        raise ValueError("Reduction must be between 0 and 100")

    savings = expenses * reduction / Decimal("100")
    new_expenses = expenses - savings

    return {
        "savings": float(savings),
        "new_expenses": float(new_expenses)
    }


def calculate_financial_metrics(revenue, expenses):
    revenue = Decimal(str(revenue))
    expenses = Decimal(str(expenses))

    if not revenue.is_finite() or not expenses.is_finite():
        raise ValueError("Values must be finite numbers")

    if revenue < 0 or expenses < 0:
        raise ValueError("Values cannot be negative")

    surplus = revenue - expenses
    margin = (
        surplus / revenue * Decimal("100")
        if revenue != 0 else Decimal("0")
    )

    return {
        "revenue": float(revenue),
        "expenses": float(expenses),
        "profit": float(surplus),
        "margin": float(margin)
    }
