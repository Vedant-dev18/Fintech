
from agent_workflow import calculate_finances, choose_financial_path

test_cases = [
    ("Loss-making business", 50000, 70000, "loss"),
    ("Low-margin business", 100000, 92000, "low_margin"),
    ("Profitable business", 100000, 70000, "profitable"),
]

for name, revenue, expenses, expected in test_cases:
    state = {
        "revenue": revenue,
        "expenses": expenses
    }

    metrics = calculate_finances(state)
    state.update(metrics)

    selected_path = choose_financial_path(state)

    print(f"\n{name}")
    print(f"Operating margin: {state['margin']:.2f}%")
    print(f"Selected path: {selected_path}")

    assert selected_path == expected, (
        f"Expected {expected}, got {selected_path}"
    )

print("\nSUCCESS: All three routing decisions are correct!")
