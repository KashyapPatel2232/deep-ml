def balance_undersample(data: list) -> list:
    """Undersample the majority classes so all classes have the same number of

    samples equal to the minority class count, preserving the original order.
    """
    if not data:
        return []

    classes_count = {}
    for sample, label in data:
        classes_count[label] = classes_count.get(label, 0) + 1

    key = min(classes_count, key=classes_count.get)
    n_min = classes_count[key]

    seen_counts = {label: 0 for label in classes_count}
    l = []

    for sample, label in data:
        if seen_counts[label] < n_min:
            l.append((sample, label))
            seen_counts[label] += 1

    return l