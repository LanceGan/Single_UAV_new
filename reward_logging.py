"""Small, dependency-free TensorBoard reward diagnostics helpers."""


def log_reward_terms(writer, terms, episode):
    total = 0.0
    for name, value in terms.items():
        numeric_value = float(value)
        writer.add_scalar(f"reward_terms/{name}", numeric_value, episode)
        total += numeric_value
    writer.add_scalar("reward_terms/total", total, episode)
